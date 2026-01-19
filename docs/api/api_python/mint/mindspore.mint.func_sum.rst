mindspore.mint.sum
==================
.. py:function:: mindspore.mint.sum(input, *, dtype=None) -> Tensor

    计算tensor所有元素的和。

    参数：
        - **input** (Tensor) - 输入tensor。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 期望输出tensor的类型。默认 ``None`` 。

    返回：
        Tensor

    .. py:function:: mindspore.mint.sum(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    计算tensor在指定维度上元素的和。

    .. note::
        Tensor类型的 `dim` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[int, tuple(int), list(int), Tensor]) - 指定计算维度。
        - **keepdim** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 期望输出tensor的类型。默认 ``None`` 。

    返回：
        Tensor
