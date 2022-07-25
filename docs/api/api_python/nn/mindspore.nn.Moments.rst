mindspore.nn.Moments
====================

.. py:class:: mindspore.nn.Moments(axis=None, keep_dims=None)

    沿指定轴 `axis` 计算输入 `x` 的均值和方差。

    参数：
        - **axis** (Union[int, tuple(int), None]) - 沿指定轴 `axis` 计算均值和方差，值为None时代表计算 `x` 所有值的均值和方差。默认值：None。
        - **keep_dims** (Union[bool, None]) - 如果为True，计算结果会保留 `axis` 的维度，即均值和方差的维度与输入的相同。如果为False或None，则会降低 `axis` 的维度。默认值：None。

    输入：
        - **x** (Tensor) - 用于计算均值和方差的任意维度的Tensor。数据类型仅支持float16和float32。

    输出：
        - **mean** (Tensor) - `x` 在 `axis` 上的均值，数据类型与输入 `x` 相同。
        - **variance** (Tensor) - `x` 在 `axis` 上的方差，数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `axis` 不是int，tuple或None。
        - **TypeError** - `keep_dims` 既不是bool也不是None。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
