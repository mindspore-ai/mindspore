mindspore.ops.FractionalAvgPool
===============================

.. py:class:: mindspore.ops.FractionalAvgPool(pooling_ratio, pseudo_random=False, overlapping=False, deterministic=False, seed=0, seed2=0)

    在输入上执行分数平均池化。

    分数平均池化类似于常规平均池化。在常规平均池中，可以通过获取集合中较小的 `N x N` 子部分（通常为2x2）的平均值来缩小输入集的大小，
    
    并尝试将集合减少N的因子，其中N是整数。分数平均池化，意味着整体缩减比N不必是整数。在每个池化区域中，执行均值运算。

    .. warning::
        `pooling_ratio` 当前只支持行和列轴，并行大于1.0，第一个和最后一个元素必须为1.0，因为我们不允许对batch和通道轴进行池化。

    参数：
        - **pooling_ratio** (list(float)) - 决定了输出的shape，数据类型是floats的列表，长度大于等于4。其值为每个维度的池化比率，目前仅支持行和列维度，
          应该大于等于0。第一个和最后一个元素必须为1.0，不支持对batch和通道轴进行池化。
        - **pseudo_random** (bool，可选) - 当设置为True时，以伪随机方式生成池序列，为False时，以随机方式生成池序列。默认为False。
          查看文章 `Fractional Max-Pooling <https://arxiv.org/pdf/1412.6071>`_ 以了解伪随机和随机之间的差异。
        - **overlapping** (bool，可选) - 当设置为True时，表示两个单元格都使用相邻池化单元边界的值进行池化，
          设置为False时，表示值不进行重复使用。默认为False。
        - **deterministic** (bool，可选) - 当设置为True时，将在计算图中的FractionalAvgPool节点上进行迭代时使用固定池区域。
          主要用于单元测试，使FractionalAvgPool具有确定性。当设置为False时，将不使用固定池区域。默认为False。
        - **seed** (int，可选) - 如果seed或seed2被设置为非零，则随机数生成器由给定的seed生成，否则，它由随机种子生成。默认为0。
        - **seed2** (int，可选) - 第二个seed，以避免发生seed碰撞。默认为0。

    输入：
        - **x** (Tensor) - 数据类型必须为：float32、float64、int32、int64。shape为 :math:`(N, H_{in}, W_{in}, C_{in})` 。

    输出：
        - **y** (Tensor) - 一个Tensor，FractionalAvgPool的输出，与 `x` 具有相同的数据类型，shape为 :math:`(N, H_{out}, W_{out}, C_{out})` 。
        - **row_pooling_sequence** (Tensor) - 一个Tensor，池边界行的结果列表，数据类型为int64。
        - **col_pooling_sequence** (Tensor) - 一个Tensor，池边界列的结果列表，数据类型为int64。

    异常：
        - **TypeError** - 如果 `x` 数据类型不是：float32、float64、int32或者int64。
        - **TypeError** - 如果 `x` 不是一个4D的Tensor。
        - **ValueError** - 如果 `x` 的元素等于0或者小于0。
        - **ValueError** - 如果 `pooling_ratio` 是一个列表，其长度不等于4。
        - **ValueError** - 如果 `pooling_ratio` 的第一个和最后一个值不等于1.0。
