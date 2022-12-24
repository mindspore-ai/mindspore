mindspore.nn.GetNextSingleOp
=============================

.. py:class:: mindspore.nn.GetNextSingleOp(dataset_types, dataset_shapes, queue_name)

    用于获取下一条数据的Cell。更详细的信息请参考 :class:`mindspore.ops.GetNext` 。

    参数：
        - **dataset_types** (list[:class:`mindspore.dtype`]) - 数据集类型。
        - **dataset_shapes** (list[tuple[int]]) - 数据集的shape。
        - **queue_name** (str) - 待获取数据的队列名称。

    输出：
        tuple[Tensor]，从数据集中获取的数据。
