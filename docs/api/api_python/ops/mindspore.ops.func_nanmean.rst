mindspore.ops.nanmean
=====================

.. py:function:: mindspore.ops.nanmean(input, axis=None, keepdims=False, *, dtype=None)

    计算 `input` 指定维度元素的平均值，忽略NaN。如果指定维度中的所有元素都是NaN，则结果将是NaN。

    参数：
        - **input** (Tensor) - 计算平均值的输入Tensor。
        - **axis** (int, 可选) - 求平均值的维度。默认值： ``None`` ，所有维度求平均值。
        - **keepdims** (bool, 可选) - 输出Tensor是否保持维度。默认值： ``False`` ，不保留维度。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出Tensor的类型。默认值： ``None`` ，输出Tensor的类型和输入一致。

    返回：
        Tensor，输入 `input` 指定维度的元素平均值，忽略NaN。

    异常：
        - **TypeError** - `input` 不是一个Tensor。
        - **TypeError** - `axis` 不是int类型。
        - **TypeError** - `keepdims` 不是bool类型。
        - **TypeError** - `dtype` 不是MindSpore的数据类型。
        - **ValueError** - `axis` 的范围不在[-r, r)，`r` 表示 `input` 的rank。
