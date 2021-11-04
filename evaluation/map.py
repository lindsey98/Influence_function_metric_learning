
import torch

def torch_all_from_dim_to_end(x, dim):
    return torch.all(x.view(*x.shape[:dim], -1), dim=-1)

def to_dtype(x, tensor=None, dtype=None):
    if not torch.is_autocast_enabled():
        dt = dtype if dtype is not None else tensor.dtype
        if x.dtype != dt:
            x = x.type(dt)
    return x

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

