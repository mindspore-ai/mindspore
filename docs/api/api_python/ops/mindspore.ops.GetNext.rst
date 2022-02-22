mindspore.ops.GetNext
======================

.. py:class:: mindspore.ops.GetNext(types, shapes, output_num, shared_name)

    返回数据集队列中的下一个元素。

    .. note::
        GetNext操作需要联网，且依赖init_dataset接口，不能单独操作。详见 `connect_network_with_dataset` 的源码。

    **参数：**

    - **types** (list[:class:`mindspore.dtype`]) - 输出的数据类型。
    - **shapes** (list[tuple[int]]) - 输出的数据大小。
    - **output_num** (int) - 输出编号、 `types` 和 `shapes` 的长度。
    - **shared_name** (str) - `init_dataset` 接口名称。

    **输入：**

    没有输入。

    **输出：**

    tuple[Tensor]，Dataset的输出。Shape和类型参见 `shapes` 、 `types` 。