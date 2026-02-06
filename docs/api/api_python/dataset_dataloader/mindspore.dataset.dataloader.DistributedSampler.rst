mindspore.dataset.dataloader.DistributedSampler
===============================================

.. py:class:: mindspore.dataset.dataloader.DistributedSampler(dataset, num_replicas=None, rank=None, shuffle=True, seed=0, drop_last=False)

    将数据集进行分片用于分布式训练的采样器。

    参数：
        - **dataset** (Dataset) - 用于采样的数据集。
        - **num_replicas** (int, 可选) - 参与分布式训练的总分片数量。默认值： ``None`` 。
        - **rank** (int, 可选) - 当前分片在 `num_replicas` 中的序列号。默认值： ``None`` 。
        - **shuffle** (bool, 可选) - 采样器是否对样本进行随机排序，默认值： ``True`` 。
        - **seed** (int, 可选) - 当设置 `shuffle` 为 ``True`` 时，用于随机排序采样器的种子值。默认值： ``0`` 。
        - **drop_last** (bool, 可选) - 采样器是否舍弃尾部数据。如果为 ``True`` ，采样器将舍弃尾部数据，使其能被等分到所有分片中；如果为 ``False`` ，采样器将添加额外索引，使数据能被等分到分片中。默认值： ``False`` 。
