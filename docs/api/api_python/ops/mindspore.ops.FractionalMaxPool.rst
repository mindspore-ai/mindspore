mindspore.ops.FractionalMaxPool
===============================

.. py:class:: mindspore.ops.FractionalMaxPool(pooling_ratio, pseudo_random=False, overlapping=False, deterministic=False, seed=0, seed2=0)

    在输入上执行分数最大池化。

    分数最大池化类似于常规最大池化。在常规最大池化中，通过取集合较小的 `N x N` 的子部分的最大值（通常为2x2）来缩小输入集的大小，目标是将集合缩小 `N` 倍，其中 `N` 为整数。
    但分数最大池化具有额外的灵活性，允许总缩小比率 `N` 为非整数值。


    .. warning::
        `pooling_ratio` 当前只支持行和列轴，并行大于1.0，第一个和最后一个元素必须为1.0，因为我们不允许对batch和通道轴进行池化。

    参数：
        - **pooling_ratio** (list(float)) - 决定了输出的shape，floats列表，长度大于等于4。对每个轴的value不能小于零0，目前仅支持行和列维度。
        - **pseudo_random** (bool，可选) - 控制序列生成机制是随机或伪随机。当设置为True时，以伪随机方式生成池序列，为False时，以随机方式生成池序列。默认值：False。
          参考 Benjamin Graham 的论文 `Fractional Max-Pooling <https://arxiv.org/pdf/1412.6071>`_ 以了解伪随机和随机池化之间的差异。
        - **overlapping** (bool，可选) - 当设置为True时，表示两个单元格都使用相邻池化单元边界的值进行池化，
          设置为False时，表示值不进行重复使用。默认值：False。
        - **deterministic** (bool，可选) - 当设置为True时，将在计算图中的FractionalMaxPool节点上进行迭代时使用固定池区域。
          主要用于单元测试，使FractionalMaxPool具有确定性。当设置为False时，将不使用固定池区域。默认值：False。
        - **seed** (int，可选) - 如果seed或seed2被设置为非零，则随机数生成器由给定的seed生成，否则，它由随机种子生成。默认值：0。
        - **seed2** (int，可选) - 第二个seed，以避免发生seed碰撞。默认值：0。

    输入：
        - **x** (Tensor) - 数据类型必须为：float32、float64、int32、int64。shape为： :math:`(N, H_{in}, W_{in}, C_{in})` 。

    输出：
        - **y** (Tensor) - 一个Tensor，FractionalMaxPool的输出，与 `x` 具有相同的数据类型，shape为： :math:`(N, H_{out}, W_{out}, C_{out})` 。
        - **row_pooling_sequence** (Tensor) - 一个Tensor，池边界行的结果列表，数据类型为int64。
        - **col_pooling_sequence** (Tensor) - 一个Tensor，池边界列的结果列表，数据类型为int64。

    异常：
        - **TypeError** - 如果 `x` 数据类型不是：float32、float64、int32或者int64。
        - **TypeError** - 如果 `x` 不是一个4D的Tensor。
        - **ValueError** - 如果 `x` 的元素等于0或者小于0。
        - **ValueError** - 如果 `pooling_ratio` 是一个列表，其长度不等于4。
        - **ValueError** - 如果 `pooling_ratio` 的第一个和最后一个值不等于1.0。
