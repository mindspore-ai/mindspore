mindspore.ops.poisson
=====================

.. py:class:: mindspore.ops.poisson(shape, mean, seed=None)

    根据泊松随机数分布生成随机数。

    .. math::
        \text{P}(i|μ) = \frac{\exp(-μ)μ^{i}}{i!}

    **参数：**
    
    - **shape** (tuple) - Tuple: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。
    - **mean** (Tensor) - 均值μ，分布参数。支持float32数据类型，应大于0。
    - **seed** (int) - 随机种子。取值须为非负数。默认值：None，等同于0。

    **返回：**
    
    Tensor，shape应与输入 `shape` 与 `mean` 进行广播之后的shape相同。数据类型支持float32。

    **异常：**
    
    - **TypeError** - `shape` 不是Tuple。
    - **TypeError** - `mean` 不是Tensor或数据类型非float32。
    - **TypeError** - `seed` 不是int类型。
