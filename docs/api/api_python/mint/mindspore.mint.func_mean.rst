mindspore.mint.mean
===================

.. py:function:: mindspore.mint.mean(input, *, dtype=None) -> Tensor

    计算tensor的均值。

    参数：
        - **input** (Tensor[Number]) - 输入tensor。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 返回的数据类型。默认 ``None`` 。

    返回：
        Tensor。


    .. py:function:: mindspore.mint.mean(input, dim, keepdim=False, *, dtype=None) -> Tensor
        :noindex:

    计算tensor在指定维度上的均值。

    .. note::
        Tensor类型的 `dim` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **input** (Tensor[Number]) - 输入tensor。
        - **dim** (Union[int, tuple(int), list(int), Tensor]) - 指定维度。
        - **keepdim** (bool) - 输出tensor是否保留维度。默认 ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 返回的数据类型。默认 ``None`` 。

    返回：
        Tensor。
