mindspore.mint.nn.SyncBatchNorm
=================================

.. py:class:: mindspore.mint.nn.SyncBatchNorm(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, process_group=None, dtype=None)

    在N维输入上进行跨设备同步批归一化（Batch Normalization，BN）。

    同步BN是跨设备同步的。相较于BN的实现仅对单个设备中的数据进行批归一化，同步BN将对指定通信组内所有设备上的输入进行批归一化。
    描述见论文 `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ 。使用mini-batch数据和学习参数进行训练，参数见如下公式。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **num_features** (int) - 指定输入Tensor的通道数量，输入Tensor的size为 :math:`(N, C, +)` 。
        - **eps** (float，可选) - :math:`\epsilon` 添加到分母中的值，以确保数值稳定。默认 ``1e-5`` 。
        - **momentum** (float，可选) - 动态均值和动态方差所使用的动量。默认 ``0.1`` 。
        - **affine** (bool，可选) - bool类型。设置为 ``True`` 时， :math:`\gamma` 和 :math:`\beta` 为可学习参数。设置为 ``False`` 时，:math:`\gamma` 和 :math:`\beta` 为不可学习参数。默认 ``True`` 。
        - **track_running_stats** (bool, 可选) - bool类型。设置为 ``True`` 时，会跟踪运行时的均值和方差。当设置为 ``False`` 时，
          则不会跟踪这些统计信息。且在train和eval模式下，该cell总是使用batch的统计信息。默认 ``True`` 。
        - **process_group** (str, 可选) - 统计数据的同步在每个进程组内单独进行。默认行为是全局同步。默认 ``None`` 。
        - **dtype** (:class:`mindspore.dtype`, 可选) - Parameters的dtype。默认 ``None`` 。

    输入：
        - **x** （Tensor） - shape为 :math:`(N, C, +)` 的Tensor。

    输出：
        Tensor，归一化后的Tensor，shape为 :math:`(N, C, +)` 。

    异常：
        - **ValueError** - `num_features` 小于1。
        - **ValueError** - `momentum` 不在范围[0, 1]内。
        - **ValueError** - `process_group` 中的rank ID不在[0, rank_size)范围内。

    样例：

    .. note::
        在运行以下示例之前，您需要配置通信环境变量。

        对于Ascend设备，用户需要准备rank table，设置rank_id和device_id。
        这里，示例使用msrun通过单个命令跨节点拉取多进程分布式任务行指令。
        请参阅 `Ascend教程
        <https://www.mindspore.cn/tutorials/en/master/parallel/msrun_launcher.html>`_
        了解更多详情。

        此示例应在多个设备上运行。