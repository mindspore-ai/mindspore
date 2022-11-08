mindspore.nn.SyncBatchNorm
===========================

.. py:class:: mindspore.nn.SyncBatchNorm(num_features, eps=1e-5, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, process_groups=None)

    在N维输入上进行跨设备同步批归一化（Batch Normalization，BN）。

    同步BN是跨设备的。BN的实现仅对每个设备中的数据进行归一化。同步BN将归一化组内的输入。描述见论文 `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ 。使用mini-batch数据和和学习参数进行训练，参数见如下公式。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. note::
        目前，SyncBatchNorm仅支持二维和四维输入。

    参数：
        - **num_features** (int) - 指定输入Tensor的通道数量，输入Tensor的size为 :math:`(N, C, H, W)` 。
        - **eps** (float) - :math:`\epsilon` 添加到分母中的值，以确保数值稳定。默认值：1e-5。
        - **momentum** (float) - 动态均值和动态方差所使用的动量。默认值：0.9。
        - **affine** (bool) - bool类型。设置为True时， :math:`\gamma` 和 :math:`\beta` 为可学习参数。默认值：True。
        - **gamma_init** (Union[Tensor, str, Initializer, numbers.Number]) - :math:`\gamma` 参数的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'、'xavier_uniform'、'he_uniform'等。默认值：'ones'。
        - **beta_init** (Union[Tensor, str, Initializer, numbers.Number]) - :math:`\beta` 参数的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'、'xavier_uniform'、'he_uniform'等。默认值：'zeros'。
        - **moving_mean_init** (Union[Tensor, str, Initializer, numbers.Number]) - 动态平均值的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'、'xavier_uniform'、'he_uniform'等。默认值：'zeros'。
        - **moving_var_init** (Union[Tensor, str, Initializer, numbers.Number]) - 动态方差的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'、'xavier_uniform'、'he_uniform'等。默认值：'ones'。
        - **use_batch_statistics** (bool) - 如果为True，则使用当前批次数据的平均值和方差值。如果为False，则使用指定的平均值和方差值。如果为None，则训练过程将使用当前批次数据的均值和方差，并跟踪动态均值和动态方差，验证过程将使用动态均值和动态方差。默认值：None。
        - **process_groups** (list) - 将设备划分为不同的同步组的列表，包含N个列表。每个列表都包含需要在同一组中同步的rank ID，其数据类型为整数且数值范围必须为[0, rank_size)并且各不相同。如果为None，表示跨所有设备同步。默认值：None。

    输入：
        - **x** （Tensor） - shape为 :math:`(N, C_{in}, H_{in}, W_{in})` 的Tensor。

    输出：
        Tensor，归一化后的Tensor，shape为 :math:`(N, C_{out}, H_{out}, W_{out})` 。

    异常：
        - **TypeError** - `num_features` 不是int。
        - **TypeError** - `eps` 不是float。
        - **TypeError** - `process_groups` 不是list。
        - **ValueError** - `num_features` 小于1。
        - **ValueError** - `momentum` 不在范围[0, 1]内。
        - **ValueError** - `process_groups` 中的rank ID不在[0, rank_size)范围内。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。