mindspore.ops.RandomPoisson
============================

.. py:class:: mindspore.ops.RandomPoisson(seed=0, seed2=0, dtype=mstype.int64)

    根据离散概率密度函数分布生成随机非负数浮点数i。函数定义如下：

    .. math::
        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    参数：
        - **seed** (int，可选) - 随机数种子。如果 `seed` 或者 `seed2` 被设置为非零，则使用这个非零值。否则使用一个随机生成的种子。默认值：0。
        - **seed2** (int，可选) - 另一个随机种子，避免发生冲突。默认值：0。
        - **dtype** (mindspore.dtype，可选) - 输出数据类型， 默人值：mstype.int64。

    输入：
        - **shape** (tuple) - 待生成的随机Tensor的shape，是一个一维Tensor。数据类型为int32或int64。
        - **rate** (Tensor) - `rate` 为Poisson分布的μ参数，决定数字的平均出现次数。数据类型是其中之一：[float16, float32, float64, int32, int64]。

    输出：
        Tensor。shape是 :math:`(*shape, *rate.shape)` ，数据类型由参数 `dtype` 指定。

    异常：
        - **TypeError** - `shape` 不是Tensor或数据类型不是int32或int64。
        - **TypeError** - `dtype` 数据类型不是int32或int64。
        - **ValueError** - `shape` 不是一维Tensor。
        - **ValueError** - `shape` 的元素存在负数。
