mindspore.ops.nansum
====================

.. py:function:: mindspore.ops.nansum(input, axis=None, keepdims=False, *, dtype=None)

    计算 `input` 指定维度元素的和，将非数字(NaNs)处理为零。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **axis** (Union[int, tuple(int)], 可选) - 求和的维度。假设 `input` 的秩为r，取值范围[-r,r)。默认值：None，对Tensor中的所有元素求和。
        - **keepdims** (bool, 可选) - 输出Tensor是否保持维度。默认值：False，不保留维度。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 输出Tensor的类型。默认值：None。

    返回：
        Tensor，输入 `input` 指定维度的元素和，将非数字(NaNs)处理为零。

        - 如果 `axis` 为None，且 `keep_dims` 为False，
          则输出一个零维Tensor，表示输入Tensor中所有元素的和。
        - 如果 `axis` 为int，值为2，并且 `keep_dims` 为False，
          则输出的shape为： :math:`(input_1, input_3, ..., input_R)` 。
        - 如果 `axis` 为tuple(int)或list(int)，值为(2, 3)，并且 `keep_dims` 为False，
          则输出的shape为 :math:`(input_1, input_4, ..., input_R)` 。

    异常：
        - **TypeError** - `input` 不是一个Tensor。
        - **TypeError** - `keepdims` 不是bool类型。
        - **TypeError** - `input` 的数据类型或 `dtype` 是complex类型。
        - **ValueError** - `axis` 的范围不在[-r, r)，r表示 `input` 的秩。
