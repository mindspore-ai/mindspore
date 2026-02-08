mindspore.mint.any
=====================

.. py:function:: mindspore.mint.any(input) -> Tensor

    检查 `input` 中是否含有 ``True``。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor

    .. py:function:: mindspore.mint.any(input, dim, keepdim=False) -> Tensor
        :noindex:

    检查 `input` 指定维度上是否含有 ``True``。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (int) - 要归约的维度。
        - **keepdim** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
