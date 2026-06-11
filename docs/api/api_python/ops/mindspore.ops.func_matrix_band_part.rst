mindspore.ops.matrix_band_part
==============================

.. py:function:: mindspore.ops.matrix_band_part(x, lower, upper)

    `mindspore.ops.matrix_band_part` 从2.9.0版本开始已被弃用，并将在未来版本中被移除。

    返回一个tensor，保留指定对角线的值，其余设为0。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **lower** (Union[int, Tensor]) - 要保留的次对角线数。如果为负数，则保留对角线下方所有元素。
        - **upper** (Union[int, Tensor]) - 要保留的超对角线数。如果为负数，则保留对角线上方所有元素。

    返回：
        Tensor
