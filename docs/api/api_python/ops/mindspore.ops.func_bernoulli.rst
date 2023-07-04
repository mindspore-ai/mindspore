mindspore.ops.bernoulli
=======================

.. py:function:: mindspore.ops.bernoulli(input, p=0.5, seed=None)

    以 `p` 的概率随机将输出的元素设置为0或1，服从伯努利分布。

    .. math::

        out_{i} \sim Bernoulli(p_{i})

    参数：
        - **input** (Tensor) - Tensor的输入，其数据类型为int8、uint8、int16、int32、int64、bool、float32或float64。
        - **p** (Union[Tensor, float], 可选) - 成功概率。 `p` 中每个值代表输出Tensor中对应位置为1的概率，如果是Tensor，其shape必须与 `input` 一致，数值范围在0到1之间。默认值： ``0.5`` 。
        - **seed** (Union[int, None], 可选) - 随机种子，用于生成随机数，数值范围为正整数。默认值： ``None`` ，表示使用时间戳。

    返回：
        - **output** (Tensor) - shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 的数据类型不在int8、uint8、int16、int32、int64、bool、float32和float64中。
        - **TypeError** - `p` 的数据类型既不是float32也不是float64。
        - **TypeError** - `seed` 不是int或None。
        - **ValueError** - `seed` 是负数。
        - **ValueError** - `p` 数值范围不在0到1之间。
        - **ValueError** - 如果 `p` 是Tensor，但是其shape与 `input` 不同。

