mindspore.ops.poisson
=====================

.. py:function:: mindspore.ops.poisson(shape, rate, seed=None, dtype=mindspore.dtype.float32)

    根据泊松随机数分布生成随机数。

    .. math::
        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    参数：
        - **shape** (Tensor) - 输出Tensor的shape，是一个一维的Tensor，其数据类型可以是mindspore.dtype.int64或者mindspore.dtype.int32。
        - **rate** (Tensor) - 泊松分布的参数均值μ，表示事件发生的概率的平均值，是一个Tensor，其数据类型可以是mindspore.dtype.int64，mindspore.dtype.int32，mindspore.dtype.float64，mindspore.dtype.float32或mindspore.dtype.float16。
        - **seed** (int) - 随机种子。取值须为非负数。默认值：None，等同于0。
        - **dtype** (mindspore.dtype) - 输出数据的数据类型。必须是mindspore.dtype.int64，mindspore.dtype.int32，mindspore.dtype.float64，mindspore.dtype.float32或mindspore.dtype.float16中的一种。默认值：mindspore.dtype.float32。

    返回：
        一个shape为`mindspore.concat([shape, mindspore.shape(mean)], axis=0)`，数据类型为`dtype`的Tensor。

    异常：
        - **TypeError** - `shape` 不是一个Tensor，或者其数据类型不是mindspore.dtype.int64或者mindspore.dtype.int32。
        - **TypeError** - `rate` 不是一个Tensor，或者数据类型不是mindspore.dtype.int64，mindspore.dtype.int32，mindspore.dtype.float64，mindspore.dtype.float32或mindspore.dtype.float16中的一种。
        - **TypeError** - `seed` 不是int类型。
        - **TypeError** - `dtype` 不是mindspore.dtype.int64，mindspore.dtype.int32，mindspore.dtype.float64，mindspore.dtype.float32或mindspore.dtype.float16中的一种。
