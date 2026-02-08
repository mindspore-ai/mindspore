mindspore.mint.nn.GroupNorm
=================================

.. py:class:: mindspore.mint.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, dtype=None)

    对mini-batch输入进行组归一化。

    Group Normalization被广泛用于递归神经网络中。适用单个训练用例的mini-batch输入归一化，详见论文 `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_ 。

    Group Normalization把通道划分为组，然后计算每一组之内的均值和方差，以进行归一化。其中 :math:`\gamma` 是通过训练学习出的scale值，:math:`\beta` 是通过训练学习出的shift值。

    公式如下，

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中， :math:`\gamma` 为 `weight`， :math:`\beta` 为 `bias`， :math:`\epsilon` 为 `eps`。

    参数：
        - **num_groups** (int) - 沿通道维度待划分的组数。
        - **num_channels** (int) - 输入的通道数。
        - **eps** (float, 可选) - 添加到分母中的值，以确保数值稳定。默认 ``1e-05`` 。
        - **affine** (bool, 可选) - 当被设置为 ``True`` 时，参数 :math:`\gamma` 和 :math:`\beta` 是可学习的。默认 ``True`` 。
        - **dtype** (:class:`mindspore.dtype`, 可选) - 参数的数据类型。默认 ``None`` 。

    输入：
        - **input** (Tensor) - :math:`(N, C, *)`，其中 :math:`*` 任意数量的附加维度。

    输出：
        - **output** (Tensor) - 被标准化和缩放偏移后的tensor，具有与 `input` 相同的shape和数据类型。

    异常：
        - **ValueError** - `num_groups` 或 `num_channels` 小于1，或 `num_channels` 未被 `num_groups` 整除。
