mindspore.ops.nan_to_num
=========================

.. py:function:: mindspore.ops.nan_to_num(x, nan=0.0, posinf=None, neginf=None)

    将 `x` 中的 `NaN` 、正无穷大和负无穷大值分别替换为 `nan`, `posinf`, 和 `neginf` 指定的值。

    参数：
        - **x** (Tensor) - shape为 :math:`(x_1, x_2, ..., x_R)` 的tensor。类型必须为float32或float16。
        - **nan** (float) - 替换 `NaN` 的值。默认值为0.0。
        - **posinf** (float) - 如果是一个数字，则为替换正无穷的值。如果为None，则将正无穷替换为 `x` 类型支持的上限。默认值为None。
        - **neginf** (float) - 如果是一个数字，则为替换负无穷的值。如果为None，则将负无穷替换为 `x` 类型支持的下限。默认值为None。

    返回：
        Tensor，数据shape和类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是一个Tensor。
        - **TypeError** - `x` 的类型既不是float16也不是float32。
