mindspore.ops.bernoulli
=======================

.. py:function:: mindspore.ops.bernoulli(x, p=0.5, seed=-1)

    以p的概率随机将输出的元素设置为0或1，服从伯努利分布。

    .. math::

        out_{i} \sim Bernoulli(p_{i})

    参数：
        - **x** (Tensor) - 任意维度的Tensor，其数据类型为int8, uint8, int16, int32，int64，bool, float32或float64。
        - **p** (Union[Tensor, float], 可选) - shape需要可以被广播到当前Tensor。其数据类型为float32或float64。`p` 中每个值代表输出Tensor中对应广播位置为1的概率，数值范围在0到1之间。默认值：0.5。
        - **seed** (int, 可选) - 随机种子，用于生成随机数，数值范围是-1或正整数。默认值：-1，代表取当前时间戳。

    返回：
        - **output** (Tensor) - shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 的数据类型不在int8、uint8、int16、int32、int64、bool、float32和float64中。
        - **TypeError** - `p` 的数据类型既不是float32也不是float64。
        - **TypeError** - `seed` 不是int。
        - **ValueError** - `seed` 是负数且不为-1。
        - **ValueError** - `p` 数值范围不在0到1之间。
