mindspore.ops.prod
==================

.. py:function:: mindspore.ops.prod(input, axis=None, keep_dims=False, dtype=None)

    返回tensor在指定轴上的乘积。

    .. warning::
        2.9.0（不含）之后版本非兼容性变更：`axis` 和 `keep_dims` 将分别改为 `dim` 和 `keepdim`，且不再支持多轴。

    参数：
        - **input** (Tensor[Number]) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor]) - 指定计算轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **keep_dims** (bool) - 输出tensor是否保留维度。默认 ``False`` 。
        - **dtype** (:class:`mindspore.dtype`) - 指定数据类型。默认 ``None`` 。

    返回：
        Tensor
