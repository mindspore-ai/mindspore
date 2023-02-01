mindspore.ops.TripletMarginLoss
===============================

.. py:class:: mindspore.ops.TripletMarginLoss(p=2, eps=1e-6, swap=False, reduction='mean')

    三元组损失函数。

    创建一个标准，用于计算输入Tensor :math:`x` 、 :math:`x2` 和 :math:`x3` 与大于 :math:`0` 的 `margin` 之间的三元组损失值。
    可以用来测量样本之间的相似度。一个三元组包含 `a` 、 `p` 和 `n` （即分别代表示  `anchor` 、 `positive examples` 和 `negative examples` ）。
    所有输入Tensor的shape都应该为 :math:`(N, D)` 。
    距离交换在V. Balntas、E. Riba等人的论文 `Learning local feature descriptors with triplets and shallow convolutional neural networks <http://158.109.8.37/files/BRP2016.pdf>`_ 中有详细的阐述。

    对于每个小批量样本，损失值为：

    .. math::
        L(a, p, n) = \max \{d(a_i, p_i) - d(a_i, n_i) + {\rm margin}, 0\}

    其中

    .. math::
        d(x_i, y_i) = \left\lVert {\bf x}_i - {\bf y}_i \right\rVert_p

    参数：        
        - **p** (int，可选) - 成对距离的范数。默认值：2。
        - **eps** (float，可选) - 默认值：1e-06。
        - **swap** (bool，可选) - 距离交换。默认值：False。
        - **reduction** (str，可选) - 指定要应用于输出的缩减。取值为"mean"、"sum"或"none"。默认值："mean"。

    输入：
        - **x** (Tensor) - 从训练集随机选取的样本。数据类型为BasicType。
        - **positive** (Tensor) - 与 `x` 为同一类的样本，数据类型与shape与 `x` 一致。
        - **negative** (Tensor) - 与 `x` 为异类的样本，数据类型与shape与 `x` 一致。
        - **margin** (Tensor) - 拉进 `a` 和 `p` 之间的距离，拉远 `a` 和 `n` 之间的距离。


    输出：
        Tensor或Scalar，如果 `reduction` 为"none"，其shape为 :math:`(N)`。否则，将返回Scalar。

    异常：
        - **TypeError** -  `x` 、 `positive` 、 `negative` 或者 `margin` 不是Tensor。
        - **TypeError** -  `x` 、 `positive` 或者 `negative` 的数据类型不是BasicType。
        - **TypeError** - `x` 、 `positive` 或者 `negative` 的数据类型不一致。
        - **TypeError** - `margin` 的数据类型不是float32。
        - **TypeError** - `p` 的数据类型不是int。
        - **TypeError** - `eps` 的数据类型不是float。
        - **TypeError** - `swap` 的数据类型不是bool。
        - **ValueError** - `x` 、 `positive` 和 `negative` 的维度同时小于等于1。
        - **ValueError** - `x` 、 `positive` 或 `negative` 的维度大于等于8。
        - **ValueError** - `margin` 的shape长度不为0。
        - **ValueError** - `x` 、 `positive` 和 `negative` 三者之间的shape无法广播。
        - **ValueError** - `reduction` 不为"mean"、"sum"或"none"。
