mindspore.dataset.dataloader.TensorDataset
==========================================

.. py:class:: mindspore.dataset.dataloader.TensorDataset(*tensors)

    由 :class:`mindspore.Tensor` 集合定义的数据集。

    每个 :class:`~mindspore.Tensor` 表示数据集的一列特征，其第0维的大小必须相同，即样本总数。
    数据集将沿着 :class:`~mindspore.Tensor` 的第0维来检索样本。

    参数：
        - **\*tensors** (mindspore.Tensor) - :class:`mindspore.Tensor` 集合。
