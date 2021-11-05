import sklearn.cluster
import sklearn.metrics.cluster
import faiss
import numpy as np

def cluster_by_kmeans(X, nb_clusters):
    """
    xs : embeddings with shape [nb_samples, nb_features]
    nb_clusters : in this case, must be equal to number of classes
    """
     # return sklearn.cluster.MiniBatchKMeans(nb_clusters, batch_size=32).fit(X).labels_
    X = X.detach().cpu().numpy()
    kmeans = faiss.Kmeans(d=X.shape[1], k=nb_clusters)
    kmeans.train(X.astype(np.float32))
    labels = kmeans.index.search(X.astype(np.float32), 1)[1]
    return np.squeeze(labels, 1)

def calc_normalized_mutual_information(ys, xs_clustered):
    return sklearn.metrics.cluster.normalized_mutual_info_score(xs_clustered, ys, average_method='geometric')


