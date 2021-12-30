from speechbrain.processing import diarization


class SpectralClustering:
  """使用spectral clustering 进行聚类
  
  spectral clustering的实现参考自speechbrain
    https://github.com/speechbrain/speechbrain
  """

  def __init__(self, affinity_type, num_speakers) -> None:
    """

    Args:
        affinity_type ([str]): 如何生成affinity matrix，cos:cosine similarity, nn: n-neighbor
        num_speakers ([int]): 说话人个数
    """

    self.num_speakers = num_speakers
    self.affinity_type = affinity_type
    if affinity_type == 'cos':
      self.cluster = diarization.Spec_Clust_unorm()
    elif affinity_type == 'nn':
      self.cluster = diarization.Spec_Cluster(
        n_clusters=num_speakers,
        assign_labels='kmeans',
        random_state=1234,
        affinity='nearest_neighbors'
      )

  def __call__(self, embeddings, n_neighbors, p_val):
    if self.affinity_type == 'cos':
      self.cluster.do_spec_clust(
        embeddings,
        self.num_speakers,
        p_val
      )
    else:
      self.cluster.perform_sc(embeddings, n_neighbors)

    labels = self.cluster.labels_
    return labels
