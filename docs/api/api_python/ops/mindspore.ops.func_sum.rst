mindspore.ops.sum
==================

.. py:function:: mindspore.ops.sum(input, dim=None, keepdim=False, *, dtype=None)

    计算tensor在指定维度上元素的和。

    .. note::
        Tensor类型的 `dim` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **dim** (Union[None, int, tuple(int), list(int), Tensor]) - 指定维度，默认 ``None`` 。
        - **keepdim** (bool) - 输出tensor是否保留维度，默认 ``False`` 。

    .. note::
        如果 `dim` 为 ``None`` ，对tensor中的所有元素求和； 如果 `dim` 为tuple, list或Tensor，将对 `dim` 中所有维度求和。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型，默认 ``None`` 。

    返回：
        Tensor
