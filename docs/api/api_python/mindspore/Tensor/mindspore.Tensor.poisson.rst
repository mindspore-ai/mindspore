mindspore.Tensor.poisson
==========================

.. py:method:: mindspore.Tensor.poisson(shape, mean, seed=0, seed2=0)

    返回与input大小相同的张量，其中每个元素都是从泊松采样的input中相应元素给出的速率参数分布。张量self的数值作为泊松分布的μ参数。

    .. math::

        \text{out}_i \sim \text{Poisson}(\text{input}_i)out*i*∼Poisson(input*i*)

    参数：
    - **shape** (tuple) - 要生成的随机张量的形状。只允许使用常量值。
    - **seed** (int, option) - 设置随机种子（0到2**32）。
    - **seed2** (int, option) - 将随机seed2设置为（0到2**32）。

    返回：
        Tensor，形状与input_Tensor相同。

    异常：
        - **TypeError** - 如果 `seed` 和 `seed2` 都不是int。
        - **TypeError** - 如果 `shape` 不是元组。
        - **TypeError** - 如果 `mean` 不是数据类型不是float32的Tensor。
