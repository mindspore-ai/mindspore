mindspore.ops.normal
======================

.. py:function:: mindspore.ops.normal(shape, mean, stddev, seed=None)

    根据正态（高斯）随机数分布生成随机数。

    参数：
        - **shape** (tuple) - Tuple: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。
        - **mean** (Union[Tensor, int, float]) - 均值μ，指定分布的峰值，数据类型支持[int8, int16, int32, int64, float16, float32]。
        - **stddev** (Union[Tensor, int, float]) - 标准差σ。大于0。数据类型支持[int8, int16, int32, int64, float16, float32]。
        - **seed** (int) - 随机种子。取值须为非负数。默认值：None，等同于0。

    返回：
        Tensor，shape应与输入 `shape` 与 `mean` 和 `stddev` 进行广播之后的shape相同。数据类型支持float32。
