mindspore.ops.group_norm
=========================

.. py:function:: mindspore.ops.group_norm(input, num_groups, weight=None, bias=None, eps=1e-5)

    在mini-batch输入上进行组归一化。

    Group Normalization被广泛用于递归神经网络中。适用单个训练用例的mini-batch输入归一化，详见论文 `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_ 。

    Group Normalization把通道划分为组，然后计算每一组之内的均值和方差，以进行归一化。其中 :math:`\gamma` 是通过训练学习出的scale值，:math:`\beta` 是通过训练学习出的shift值。

    公式如下，

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中， :math:`\gamma` 为 `weight`， :math:`\beta` 为 `bias`， :math:`\epsilon` 为 `eps`。

    参数：
        - **input** (Tensor) - shape为 :math:`(N, C, *)` 的特征输入，其中 :math:`*` 表示任意的附加维度。
        - **num_groups** (int) - 沿通道维度待划分的组数。
        - **weight** (Tensor, 可选) - shape为 :math:`(C,)` ，默认值为： ``None`` ，具有与 `input` 相同的数据类型。
        - **bias** (Tensor, 可选) - shape为 :math:`(C,)` ，默认值为： ``None`` ，具有与 `input` 相同的数据类型。
        - **eps** (float, 可选) - 添加到分母中的值，以确保数值稳定。默认值： ``1e-5`` 。

    返回：
        Tensor，标准化和缩放的偏移Tensor，具有与 `input` 相同的shape和数据类型。

    异常：
        - **TypeError** - `num_groups` 不是int。
        - **TypeError** - `eps` 不是float。
        - **ValueError** - `num_groups` 小于1。
        - **ValueError** - `C` ( `input` 的第二维) 未被 `num_groups` 整除。