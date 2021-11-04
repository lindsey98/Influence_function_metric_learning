
import torch
import faiss
import logging
import numpy as np

def torch_all_from_dim_to_end(x, dim):
    return torch.all(x.view(*x.shape[:dim], -1), dim=-1)

def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x

def is_list_or_tuple(x):
    return isinstance(x, (list, tuple))

def to_numpy(v):
    if is_list_or_tuple(v):
        return np.stack([to_numpy(sub_v) for sub_v in v], axis=1)
    try:
        return v.cpu().numpy()
    except AttributeError:
        return v

def to_device(x, tensor=None, device=None, dtype=None):
    dv = device if device is not None else tensor.device
    if x.device != dv:
        x = x.to(dv)
    if dtype is not None:
        x = to_dtype(x, dtype=dtype)
    return x

def add_to_index_and_search(index, reference_embeddings, test_embeddings, k):
    index.add(reference_embeddings)
    return index.search(test_embeddings, k)

def try_gpu(cpu_index, reference_embeddings, test_embeddings, k):
    # https://github.com/facebookresearch/faiss/blob/master/faiss/gpu/utils/DeviceDefs.cuh
    gpu_index = None
    gpus_are_available = faiss.get_num_gpus() > 0
    if gpus_are_available:
        max_k_for_gpu = 1024 if float(torch.version.cuda) < 9.5 else 2048
        if k <= max_k_for_gpu:
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index)
    try:
        return add_to_index_and_search(
            gpu_index, reference_embeddings, test_embeddings, k
        )
    except (AttributeError, RuntimeError) as e:
        if gpus_are_available:
            logging.warning(
                f"Using CPU for k-nn search because k = {k} > {max_k_for_gpu}, which is the maximum allowable on GPU."
            )
        return add_to_index_and_search(
            cpu_index, reference_embeddings, test_embeddings, k
)

def maybe_get_avg_of_avgs(accuracy_per_sample, sample_labels, avg_of_avgs):
    if avg_of_avgs:
        unique_labels = torch.unique(sample_labels, dim=0)
        mask = torch_all_from_dim_to_end(
            sample_labels == unique_labels.unsqueeze(1), 2
        )
        mask = torch.t(mask)
        acc_sum_per_class = torch.sum(accuracy_per_sample.unsqueeze(1) * mask, dim=0)
        mask_sum_per_class = torch.sum(mask, dim=0)
        average_per_class = acc_sum_per_class / mask_sum_per_class
        return torch.mean(average_per_class).item()
    return torch.mean(accuracy_per_sample).item()


def get_relevance_mask(
    shape,
    gt_labels,
    embeddings_come_from_same_source,
    label_counts,
    label_comparison_fn,
):
    relevance_mask = torch.zeros(size=shape, dtype=torch.bool, device=gt_labels.device)

    for label, count in zip(*label_counts):
        matching_rows = torch.where(
            torch_all_from_dim_to_end(gt_labels == label, 1)
        )[0]
        max_column = count - 1 if embeddings_come_from_same_source else count
        relevance_mask[matching_rows, :max_column] = True
    return relevance_mask



def mean_average_precision(
    knn_labels,
    gt_labels,
    embeddings_come_from_same_source,
    avg_of_avgs,
    label_comparison_fn,
    relevance_mask=None,
    at_r=False,
):
    device = gt_labels.device
    num_samples, num_k = knn_labels.shape[:2]
    relevance_mask = (
        torch.ones((num_samples, num_k), dtype=torch.bool, device=device)
        if relevance_mask is None
        else relevance_mask
    )

    is_same_label = label_comparison_fn(gt_labels, knn_labels)
    equality = is_same_label * relevance_mask
    cumulative_correct = torch.cumsum(equality, dim=1)
    k_idx = torch.arange(1, num_k + 1, device=device).repeat(num_samples, 1)
    precision_at_ks = (
        to_dtype(cumulative_correct * equality, dtype=torch.float64) / k_idx
    )

    summed_precision_per_row = torch.sum(precision_at_ks * relevance_mask, dim=1)
    if at_r:
        max_possible_matches_per_row = torch.sum(relevance_mask, dim=1)
    else:
        max_possible_matches_per_row = torch.sum(equality, dim=1)
        max_possible_matches_per_row[max_possible_matches_per_row == 0] = 1
    accuracy_per_sample = summed_precision_per_row / max_possible_matches_per_row

    return maybe_get_avg_of_avgs(accuracy_per_sample, gt_labels, avg_of_avgs)


def get_label_match_counts(query_labels, reference_labels):
    unique_query_labels = torch.unique(query_labels, dim=0) # get unique query labels
    comparison = unique_query_labels[:, None] == reference_labels # get counts of each query label in reference set
    match_counts = torch.sum(torch_all_from_dim_to_end(comparison, 2), dim=1)

    return (unique_query_labels, match_counts) # tuple of class label and its count

def get_lone_query_labels(
    query_labels,
    label_counts,
    embeddings_come_from_same_source,
    label_comparison_fn,
):
    unique_labels, match_counts = label_counts
    if embeddings_come_from_same_source: # if query set = reference (gallery) set
        label_matches_itself = label_comparison_fn(unique_labels, unique_labels)
        lone_condition = (
            match_counts - to_dtype(label_matches_itself, dtype=torch.long) <= 0
        )
    else:
        lone_condition = match_counts == 0

    lone_query_labels = unique_labels[lone_condition] # get uncovered class in reference set
    if len(lone_query_labels) > 0:
        comparison = query_labels[:, None] == lone_query_labels # those query samples do not have same-label samples in reference set, shall ignore
        not_lone_query_mask = ~torch.any(
            torch_all_from_dim_to_end(comparison, 2), dim=1
        )
    else:
        not_lone_query_mask = torch.ones(
            query_labels.shape[0], dtype=torch.bool, device=query_labels.device
        ) # all query set classes are seen in reference set
    return lone_query_labels, not_lone_query_mask

def mean_average_precision_at_r(
    knn_labels,
    gt_labels,
    embeddings_come_from_same_source,
    label_counts,
    avg_of_avgs,
    label_comparison_fn,
):
    relevance_mask = get_relevance_mask(
        knn_labels.shape[:2],
        gt_labels,
        embeddings_come_from_same_source,
        label_counts,
        label_comparison_fn,
    )
    return mean_average_precision(
        knn_labels,
        gt_labels,
        embeddings_come_from_same_source,
        avg_of_avgs,
        label_comparison_fn,
        relevance_mask=relevance_mask,
        at_r=True,
    )

def determine_k(num_reference_embeddings, embeddings_come_from_same_source):
    self_count = int(embeddings_come_from_same_source)
    return num_reference_embeddings - self_count

# modified from https://github.com/facebookresearch/deepcluster
def get_knn(
    reference_embeddings, test_embeddings, k, embeddings_come_from_same_source=False
):
    if embeddings_come_from_same_source:
        k = k + 1
    device = reference_embeddings.device
    reference_embeddings = to_numpy(reference_embeddings).astype(np.float32)
    test_embeddings = to_numpy(test_embeddings).astype(np.float32)

    d = reference_embeddings.shape[1]
    logging.info("running k-nn with k=%d" % k)
    logging.info("embedding dimensionality is %d" % d)
    cpu_index = faiss.IndexFlatL2(d)
    distances, indices = try_gpu(cpu_index, reference_embeddings, test_embeddings, k)
    distances = to_device(torch.from_numpy(distances), device=device)
    indices = to_device(torch.from_numpy(indices), device=device)
    if embeddings_come_from_same_source:
        return indices[:, 1:], distances[:, 1:]
    return indices, distances