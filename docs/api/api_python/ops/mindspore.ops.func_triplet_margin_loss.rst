mindspore.ops.triplet_margin_loss
==================================

.. py:function:: mindspore.ops.triplet_margin_loss(anchor, positive, negative, margin=1.0, p=2, eps=1e-06, swap=False, reduction='mean')

    三元组损失函数。
    详情请查看 :class:`mindspore.nn.TripletMarginLoss` 。

    参数：
        - **anchor** (Tensor) - 从训练集随机选取的样本。数据类型为BasicType。
        - **positive** (Tensor) - 与 `anchor` 为同一类的样本，数据类型与shape与 `anchor` 一致。
        - **negative** (Tensor) - 与 `anchor` 为异类的样本，数据类型与shape与 `anchor` 一致。
        - **margin** (float，可选) - 用于拉进 `anchor` 和 `positive` 之间的距离，拉远 `anchor` 和 `negative` 之间的距离。默认值： ``1.0`` 。
        - **p** (int，可选) - 成对距离的范数。默认值： ``2`` 。
        - **eps** (float，可选) - 防止除数为 0。默认值： ``1e-06`` 。
        - **swap** (bool，可选) - 距离交换。默认值： ``False`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    返回：
        Tensor。如果 `reduction` 为"none"，其shape为 :math:`(N)`。否则，将返回Scalar。

    异常：
        - **TypeError** - `anchor` 、 `positive` 或者 `negative` 不是Tensor。
        - **TypeError** - `anchor` 、 `positive` 或者 `negative` 的数据类型不一致。
        - **TypeError** - `margin` 的数据类型不是float。
        - **TypeError** - `p` 的数据类型不是int。
        - **TypeError** - `eps` 的数据类型不是float。
        - **TypeError** - `swap` 的数据类型不是bool。
        - **ValueError** - `anchor` 、 `positive` 和 `negative` 的维度同时小于等于1。
        - **ValueError** - `anchor` 、 `positive` 或 `negative` 的维度大于等于8。
        - **ValueError** - `anchor` 、 `positive` 和 `negative` 三者之间的shape无法广播。
        - **ValueError** - `reduction` 不为"mean"、"sum"或"none"。