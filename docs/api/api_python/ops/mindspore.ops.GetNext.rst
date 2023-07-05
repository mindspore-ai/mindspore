mindspore.ops.GetNext
======================

.. py:class:: mindspore.ops.GetNext(types, shapes, output_num, shared_name)

    返回数据集队列中的下一个元素。

    .. note::
        GetNext操作需要与network一起使用，且依赖'dataset'接口，例如： :class:`mindspore.dataset.MnistDataset` 。不能单独操作。详见 :class:`mindspore.connect_network_with_dataset` 的源码。

    参数：
        - **types** (list[:class:`mindspore.dtype`]) - 输出的数据类型。
        - **shapes** (list[tuple[int]]) - 输出数据的shape大小。
        - **output_num** (int) - 输出编号、 `types` 和 `shapes` 的长度。
        - **shared_name** (str) - 待获取数据的队列名称。

    输入：
        没有输入。

    输出：
        tuple[Tensor]，Dataset的输出。Shape和类型参见 `shapes` 、 `types` 。