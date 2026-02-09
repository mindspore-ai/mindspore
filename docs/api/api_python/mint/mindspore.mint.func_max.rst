mindspore.mint.max
===================

.. py:function:: mindspore.mint.max(input) -> Tensor

    返回tensor的最大值。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor

    .. py:function:: mindspore.mint.max(input, dim, keepdim=False)
        :noindex:

    返回tensor在指定维度上的最大值及其索引。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 指定计算维度。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False``。

    返回：
        两个tensor组成的tuple(max, max_indices)。

    .. py:function:: mindspore.mint.max(input, other) -> Tensor
        :noindex:

    详情请参考 :func:`mindspore.mint.maximum`。
