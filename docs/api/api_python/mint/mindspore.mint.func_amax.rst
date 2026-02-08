mindspore.mint.amax
========================

.. py:function:: mindspore.mint.amax(input, dim=(), keepdim=False)

    返回tensor在指定维度上的最大值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[int, tuple(int), list(int)], 可选) - 指定计算维度。
          当 `dim` 为 `()` 时，计算 `input` 中的所有元素。默认 ``()``。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False``。

    返回：
        Tensor

