mindspore.ops.uniform
=======================

.. py:function:: mindspore.ops.uniform(shape, minval, maxval, seed=None, dtype=mstype.float32)

    生成服从均匀分布的随机数。

    .. note::
        广播后，任意位置上Tensor的最小值都必须小于最大值。

    参数：
        - **shape** (Union[tuple, Tensor]) - 指定输出shape，任意维度的Tensor。
        - **minval** (Tensor) - 指定生成随机值的最小值，其数据类型为int32或float32。如果数据类型为int32，则只允许输入一个数字。
        - **maxval** (Tensor) - 指定生成随机值的最大值，其数据类型为int32或float32。如果数据类型为int32，则只允许输入一个数字。
        - **seed** (int) - 指定随机种子，用于随机数生成器生成伪随机数。随机数为非负数。默认值： ``None`` （将被视为0）。
        - **dtype** (mindspore.dtype) - 指定输入的数据类型。如果数据类型为int32，则从离散型均匀分布中生成数值型数据；如果数据类型是float32，则从连续型均匀分布中生成数值型数据。仅支持这两种数据类型。默认值： ``mstype.float32`` 。

    返回：
        Tensor，shape等于输入 `shape` 与 `minval` 和 `maxval` 广播后的shape。数据类型由输入 `dtype` 决定。

    异常：
        - **TypeError** - `shape` 不是tuple或Tensor。
        - **TypeError** - `minval` 或 `maxval` 的数据类型既不是int32，也不是float32，并且 `minval` 的数据类型与 `maxval` 的不同。
        - **TypeError** - `seed` 不是int。
        - **TypeError** - `dtype` 既不是int32，也不是float32。