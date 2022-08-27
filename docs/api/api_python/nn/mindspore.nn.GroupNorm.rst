mindspore.nn.GroupNorm
=======================

.. py:class:: mindspore.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, gamma_init='ones', beta_init='zeros')

    在mini-batch输入上进行组归一化。

    Group Normalization被广泛用于递归神经网络中。适用单个训练用例的mini-batch输入归一化，详见论文 `Group Normalization <https://arxiv.org/pdf/1803.08494.pdf>`_ 。

    Group Normalization把通道划分为组，然后计算每一组之内的均值和方差，以进行归一化。

    公式如下，

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    参数：
        - **num_groups** (int) - 沿通道维度待划分的组数。
        - **num_channels** (int) - 输入的通道数。
        - **eps** (float) - 添加到分母中的值，以确保数值稳定。默认值：1e-5。
        - **affine** (bool) - Bool类型，当设置为True时，给该层添加可学习的仿射变换参数，即gama与beta。默认值：True。
        - **gamma_init** (Union[Tensor, str, Initializer, numbers.Number]) - gamma参数的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'、'xavier_uniform'、'he_uniform'等。默认值：'ones'。如果gamma_init是Tensor，则shape必须为 :math:`(num\_channels)` 。
        - **beta_init** (Union[Tensor, str, Initializer, numbers.Number]) - beta参数的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'、'xavier_uniform'、'he_uniform'等。默认值：'zeros'如果gamma_init是Tensor，则shape必须为 :math:`(num\_channels)` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, H, W)` 的特征输入。

    输出：
        Tensor，标准化和缩放的偏移Tensor，具有与 `x` 相同的shape和数据类型。

    异常：
        - **TypeError** - `num_groups` 或 `num_channels` 不是int。
        - **TypeError** - `eps` 不是float。
        - **TypeError** - `affine` 不是bool。
        - **ValueError** - `num_groups` 或 `num_channels` 小于1。
        - **ValueError** - `num_channels` 未被 `num_groups` 整除。