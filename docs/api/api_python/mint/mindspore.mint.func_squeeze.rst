mindspore.mint.squeeze
==========================

.. py:function:: mindspore.mint.squeeze(input, dim)

    返回删除指定 `dim` 中大小为1的维度后的Tensor。

    如果 :math:`dim=()` ，则删除所有大小为1的维度。
    如果指定了 `dim`，删除指定 `dim` 中大小为1的维度。
    例如，如果不指定维度，即 :math:`dim=()` ，输入的shape为(A, 1, B, C, 1, D)，则输出的Tensor的shape为(A, B, C, D)；如果指定维度，那么squeeze操作仅在指定维度中进行。
    如果输入的shape为(A, 1, B)， :math:`axis=0` 或 :math:`axis=2` 时不会改变输入的Tensor，但 :math:`dim=1` 时，会使输入Tensor的shape变为(A, B)。

    .. note::
        - 请注意，在动态图模式下，输出Tensor将与输入Tensor共享数据，并且没有Tensor数据复制过程。
        - 维度索引从0开始，并且必须在 `[-input.ndim, input.ndim)` 范围内。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 用于计算Squeeze的输入Tensor。其shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dim** (Union[int, tuple(int)]) - 指定待删除shape的维度索引。其会删除给定 `dim` 参数中所有大小为1的维度。如果指定了维度索引，其数据类型必须为int32或int64。

    返回：
        Tensor，shape为 :math:`(x_1, x_2, ..., x_S)` 。

    异常：
        - **TypeError** - `input` 不是tensor。
        - **TypeError** - `dim` 不是int、tuple。
        - **TypeError** - `dim` 是tuple，但其元素不全是int。