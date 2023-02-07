mindspore.nn.BatchNorm2d
=========================

.. py:class:: mindspore.nn.BatchNorm2d(num_features, eps=1e-5, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')

    在四维输入（具有额外通道维度的小批量二维输入）上应用批归一化处理（Batch Normalization Layer），以避免内部协变量偏移。批归一化广泛应用于卷积网络中。请见论文 `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ 。使用mini-batch数据和学习参数进行训练，这些参数见以下公式：

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. note::
        BatchNorm的实现在图模式和PyNative模式下是不同的，因此不建议在网络初始化后更改其模式。

        需要注意的是，更新 :math:`moving\_mean` 和 :math:`moving\_var` 的公式为：

        .. math::
            \text{moving_mean}=\text{moving_mean*momentum}+μ_β\text{*(1−momentum)}\\
            \text{moving_var}=\text{moving_var*momentum}+σ^2_β\text{*(1−momentum)}

        其中， :math:`moving\_mean` 是更新后的均值， :math:`moving\_var` 是更新后的方差， :math:`μ_β, σ^2_β` 是每一批的数据的观测值(均值和方差)。

    参数：
        - **num_features** (int) - 通道数量，输入Tensor shape :math:`(N, C, H, W)` 中的 `C` 。
        - **eps** (float) - :math:`\epsilon` 加在分母上的值，以确保数值稳定。默认值：1e-5。
        - **momentum** (float) - 动态均值和动态方差所使用的动量。默认值：0.9。
        - **affine** (bool) - bool类型。设置为True时，可学习 :math:`\gamma` 和 :math:`\beta` 值。默认值：True。
        - **gamma_init** (Union[Tensor, str, Initializer, numbers.Number]) - :math:`\gamma` 参数的初始化方法。str的值引用自函数 `mindspore.common.initializer`，包括'zeros'、'ones'等。默认值：'ones'。
        - **beta_init** (Union[Tensor, str, Initializer, numbers.Number]) - :math:`\beta` 参数的初始化方法。str的值引用自函数 `mindspore.common.initializer`，包括'zeros'、'ones'等。默认值：'zeros'。
        - **moving_mean_init** (Union[Tensor, str, Initializer, numbers.Number]) - 动态平均值的初始化方法。str的值引用自函数 `mindspore.common.initializer`，包括'zeros'、'ones'等。默认值：'zeros'。
        - **moving_var_init** (Union[Tensor, str, Initializer, numbers.Number]) - 动态方差的初始化方法。str的值引用自函数 `mindspore.common.initializer`，包括'zeros'、'ones'等。默认值：'ones'。
        - **use_batch_statistics** (bool) - 

          - 如果为True，则使用当前批处理数据的平均值和方差值，并跟踪运行平均值和运行方差。
          - 如果为False，则使用指定值的平均值和方差值，不跟踪统计值。
          - 如果为None，则根据训练和验证模式自动设置 `use_batch_statistics` 为True或False。在训练时， `use_batch_statistics会` 设置为True。在验证时， `use_batch_statistics` 会自动设置为False。默认值：None。
        - **data_format** (str) - 数据格式可为'NHWC'或'NCHW'。默认值：'NCHW'。

    输入：
        - **x** (Tensor) - 输入shape为 :math:`(N, C, H, W)` 的Tensor。

    输出：
        Tensor，归一化后的Tensor，shape为 :math:`(N, C, H, W)` 。

    异常：
        - **TypeError** - `num_features` 不是整数。
        - **TypeError** - `eps` 不是浮点数。
        - **ValueError** - `num_features` 小于1。
        - **ValueError** - `momentum` 不在范围[0, 1]内。
        - **ValueError** - `data_format` 既不是'NHWC'也不是'NCHW'。
