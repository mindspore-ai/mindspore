mindspore.nn.TripletMarginLoss
===============================

.. py:class:: mindspore.nn.TripletMarginLoss(p=2, swap=False, eps=1e-06, reduction='mean')

    执行三元组损失函数的操作。

    三元组损失值通常用来测量样本之间的相似度，由一个三元组和一个大于 :math:`0` 的 :math:`margin` 计算得到。
    其中，三元组由下面公式中的 :math:`a` 、 :math:`p` 和 :math:`n` 组成。

    所有输入Tensor的shape都应该为 :math:`(N, *)` ，其中 :math:`N` 代表批处理的数量， :math:`*` 代表任意数量的附加维度。
    距离交换在V. Balntas、E. Riba等人在论文 `Learning local feature descriptors with triplets and shallow convolutional neural networks <http://158.109.8.37/files/BRP2016.pdf>`_ 中有详细的阐述。

    对于每个小批量样本，损失值为：

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}

    其中

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    参数：
        - **p** (int，可选) - 成对距离的范数。默认值：2。
        - **swap** (bool，可选) - 距离交换。默认值：False。
        - **eps** (float，可选) - 防止除数为 0。默认值：1e-06。
        - **reduction** (str，可选) - 指定要应用于输出的缩减方式，取值为"mean"、"sum"或"none"。默认值："mean"。

    输入：
        - **x** (Tensor) - 从训练集随机选取的样本。数据类型为BasicType。即上述公式中的 :math:`a` 。
        - **positive** (Tensor) - 与 `x` 为同一类的样本，数据类型与shape与 `x` 一致。即上述公式中的 :math:`p` 。
        - **negative** (Tensor) - 与 `x` 为异类的样本，数据类型与shape与 `x` 一致。即上述公式中的 :math:`n` 。
        - **margin** (Union[Tensor, float]) - 用于拉进 `x` 和 `positive` 之间的距离，拉远 `x` 和 `negative` 之间的距离。

    输出：
        Tensor。如果 `reduction` 为"none"，其shape为 :math:`(N)`。否则，将返回Scalar。

    异常：
        - **TypeError** - `x` 、 `positive` 、 `negative` 不是Tensor。
        - **TypeError** - `x` 、 `positive` 或者 `negative` 的数据类型不一致。
        - **TypeError** - `p` 的数据类型不是int。
        - **TypeError** - `eps` 的数据类型不是float。
        - **TypeError** - `swap` 的数据类型不是bool。
        - **ValueError** - `x` 、 `positive` 和 `negative` 的维度同时小于等于1。
        - **ValueError** - `x` 、 `positive` 或 `negative` 的维度大于等于8。
        - **ValueError** - `margin` 的shape长度不为0。
        - **ValueError** - `x` 、 `positive` 和 `negative` 三者之间的shape无法广播。
        - **ValueError** - `reduction` 不为"mean"、"sum"或"none"。