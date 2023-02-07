mindspore.ops.nansum
====================

.. py:function:: mindspore.ops.nansum(x, axis=None, keepdims=False, *, dtype=None)

    计算 `x` 指定维度元素的和，将非数字(NaNs)处理为零。

    参数：
        - **x** (Tensor) - 输入Tensor。
        - **axis** (Union[int, tuple(int)], 可选) - 求和的维度。假设 `x` 的秩为r，取值范围[-r,r)。默认值：None，对Tensor中的所有元素求和。
        - **keepdims** (bool, 可选) - 输出Tensor是否保持维度。默认值：False，不保留维度。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出Tensor的类型。默认值：None。

    返回：
        Tensor，输入 `x` 指定维度的元素和，将非数字(NaNs)处理为零。

        - 如果 `axis` 为None，且 `keep_dims` 为False，
          则输出一个零维Tensor，表示输入Tensor中所有元素的和。
        - 如果 `axis` 为int，值为2，并且 `keep_dims` 为False，
          则输出的shape为： :math:`(x_1, x_3, ..., x_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，值为(2, 3)，并且 `keep_dims` 为False，
          则输出的shape为 :math:`(x_1, x_4, ..., x_R)` 。

    异常：
        - **TypeError** - `x` 不是一个Tensor。
        - **TypeError** - `keepdims` 不是bool类型。
        - **TypeError** - `x` 的数据类型或 `dtype` 是complex类型。
        - **ValueError** - `axis` 的范围不在[-r, r)，r表示 `x` 的秩。
