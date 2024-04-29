mindspore.mint.normal
======================

.. py:function:: mindspore.mint.normal(mean, std, generator=None)

    根据正态（高斯）随机数分布生成随机数。

    参数：
        - **mean** (Union[float, Tensor]) - 均值μ，指定分布的峰值。
        - **std** (Union[float, Tensor]) - 标准差σ。大于0。
        - **generator** (Generator) - 管理随机数状态的生成器。默认值： ``None`` 。当 `generator` 是 ``None`` 时，随机函数会使用默认生成器。详情请参见：:class:`mindspore.nn.Generator` 。

    返回：
        Tensor, shape应与 `mean` 和 `std` 进行广播之后的shape相同。数据类型支持[float32, float64]。
