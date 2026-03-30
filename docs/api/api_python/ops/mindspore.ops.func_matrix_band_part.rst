mindspore.ops.matrix_band_part
==============================

.. py:function:: mindspore.ops.matrix_band_part(x, lower, upper)

    返回一个tensor，保留指定对角线的值，其余设为0。

    .. warning::
        - 该接口已废弃，2.9.0（不含）之后版本将删除。
        - 这是一个实验性API，后续可能修改或删除。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **lower** (Union[int, Tensor]) - 要保留的次对角线数。如果为负数，则保留对角线下方所有元素。
        - **upper** (Union[int, Tensor]) - 要保留的超对角线数。如果为负数，则保留对角线上方所有元素。

    返回：
        Tensor
