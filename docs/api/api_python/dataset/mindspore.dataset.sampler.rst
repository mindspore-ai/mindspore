.. list-table:: 参数 `sampler` 和 `num_samples` ， `shuffle` ， `num_shards` ， `shard_id` 的不同组合得到的采样器
    :widths: 150 150 50 50 350
    :header-rows: 1

    * - 参数 `sampler`
      - 参数 `num_shards` / `shard_id`
      - 参数 `shuffle`
      - 参数 `num_samples`
      - **使用的采样器**
    * - `mindspore.dataset.Sampler` 类型
      - *None*
      - *None*
      - *None*
      - **sampler**
    * - `numpy.ndarray,list,tuple,int` 类型
      - /
      - /
      - *num_samples*
      - *SubsetSampler(indices =* **sampler** *, num_samples =* **num_samples** *)*
    * - `iterable` 类型
      - /
      - /
      - *num_samples*
      - *IterSampler(sampler =* **sampler** *, num_samples =* **num_samples** *)*
    * - *None*
      - *num_shards* / *shard_id*
      - *None* / *True*
      - *num_samples*
      - *DistributedSampler(num_shards =* **num_shards** *, shard_id =* **shard_id** *, shuffle =* **True** *, num_samples =* **num_samples** *)*
    * - *None*
      - *num_shards* / *shard_id*
      - *False*
      - *num_samples*
      - *DistributedSampler(num_shards =* **num_shards** *, shard_id =* **shard_id** *, shuffle =* **False** *, num_samples =* **num_samples** *)*
    * - *None*
      - *None*
      - *None* / *True*
      - *None*
      - *RandomSampler(num_samples =* **num_samples** *)*
    * - *None*
      - *None*
      - *None* / *True*
      - *num_samples*
      - *RandomSampler(replacement =* **True** *, num_samples =* **num_samples** *)*
    * - *None*
      - *None*
      - *False*
      - *num_samples*
      - *SequentialSampler(num_samples =* **num_samples** *)*
