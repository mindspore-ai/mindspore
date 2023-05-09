mindspore.ops.random_poisson
============================

.. py:function:: mindspore.ops.random_poisson(shape, rate, seed=None, dtype=mstype.float32)

    从一个指定均值为 `rate` 的泊松分布中，随机生成形状为 `shape` 的随机数Tensor。

    .. math::

        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    参数：
        - **shape** (Tensor) - 表示要从每个分布中采样的随机数张量的形状。必须是一个一维的张量且数据类型必须是 `mindspore.dtype.int32` 或者 `mindspore.dtype.int64` 。
        - **rate** (Tensor) - 泊松分布的 :math:`μ` 参数，表示泊松分布的均值，同时也是分布的方差。必须是一个张量，且其数据类型必须是以下类型中的一种：mindspore.dtype.int64，mindspore.dtype.int32，mindspore.dtype.float64，mindspore.dtype.float32或者mindspore.dtype.float16。
        - **seed** (int, 可选) - 随机数种子，用于在随机数引擎中产生随机数。必须是一个非负的整数，``None`` 表示使用 ``0`` 作为随机数种子。默认值：``None`` 。
        - **dtype** (mindspore.dtype) - 表示要生成的随机数张量的数据类型。必须是mindspore.dtype类型，可以是以下值中的一种：mindspore.dtype.int64，mindspore.dtype.int32，mindspore.dtype.float64，mindspore.dtype.float32或者mindspore.dtype.float16。默认值：``mstype.float32`` 。

    返回：
        返回一个张量，它的形状由入参 `shape` 和 `rate` 共同决定： `mindspore.concat(['shape', mindspore.shape('rate')], axis=0)` ，它的数据类型由入参 `dtype` 决定。

    异常：
        - **TypeError** - 如果 `shape` 不是一个张量。
        - **TypeError** - 如果 `shape` 张量的数据类型不是mindspore.dtype.int64或mindspore.dtype.int32。
        - **ValueError** - 如果 `shape` 张量的形状不是一维的。
        - **TypeError** - 如果 `rate` 不是一个张量。
        - **TypeError** - 如果 `rate` 张量的数据类型不是mindspore.dtype.int64，mindspore.dtype.int32，mindspore.dtype.float64，mindspore.dtype.float32或者mindspore.dtype.float16。
        - **TypeError** - 如果 `seed` 不是一个非负整型。
        - **TypeError** - 如果 `dtype` 不是mindspore.dtype.int64，mindspore.dtype.int32，mindspore.dtype.float64，mindspore.dtype.float32或者mindspore.dtype.float16。
        - **ValueError** - 如果 `shape` 张量中有非正数。
