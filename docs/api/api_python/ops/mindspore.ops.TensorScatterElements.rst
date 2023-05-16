mindspore.ops.TensorScatterElements
===================================

.. py:class:: mindspore.ops.TensorScatterElements(axis=0, reduction="none")

    根据指定的规约算法逐元素更新输入Tensor的值。

    更多参考相见 :func:`mindspore.ops.tensor_scatter_elements`。

    .. warning::
        如果 `indices` 中有多个索引向量对应于同一位置，则输出中该位置值是不确定的。

    参数：
        - **axis** (int，可选) - 指定进行操作的轴。默认值： ``0`` 。
        - **reduction** (str，可选) - 指定进行的reduction操作。默认值是"none"，可选"add"。

    输入：
        - **data** (Tensor) - 输入Tensor。 其rank必须至少为1。
        - **indices** (Tensor) - 输入Tensor的索引，数据类型为int32或int64的。其rank必须和 `data` 一致。取值范围是[-s, s)，这里的s是 `data` 在 `axis` 指定轴的size。
        - **updates** (Tensor) - 指定与 `data` 进行reduction操作的Tensor，其数据类型和shape与 `data` 相同。

    输出：
        Tensor，shape和数据类型与输入 `data` 相同。
