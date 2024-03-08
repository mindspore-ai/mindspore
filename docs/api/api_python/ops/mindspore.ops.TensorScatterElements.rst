mindspore.ops.TensorScatterElements
===================================

.. py:class:: mindspore.ops.TensorScatterElements(axis=0, reduction="none")

    将 `updates` 中所有的元素按照 `reduction` 指定的归约操作写入 `input_x` 中 `indices` 指定的索引处。
    `axis` 控制scatter操作的方向。

    更多参考相见 :func:`mindspore.ops.tensor_scatter_elements`。

    .. warning::
        如果 `indices` 中有多个索引向量对应于同一位置，则输出中该位置值是不确定的。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **axis** (int，可选) - 指定进行操作的轴。默认值： ``0`` 。
        - **reduction** (str，可选) - 指定进行的reduction操作。默认值是 ``"none"`` ，可选 ``"add"`` 。

    输入：
        - **data** (Tensor) - 输入Tensor。 其rank必须至少为1。
        - **indices** (Tensor) - `data` 执行scatter操作的目标索引，数据类型为int32或int64。其rank必须和 `data` 一致。取值范围是[-s, s)，s是 `data` 在 `axis` 指定轴的size。
        - **updates** (Tensor) - 指定与 `data` 进行scatter操作的Tensor，其数据类型与 `data` 类型相同，shape与 `indices` 的shape相同。

    输出：
        Tensor，shape和数据类型与输入 `data` 相同。
