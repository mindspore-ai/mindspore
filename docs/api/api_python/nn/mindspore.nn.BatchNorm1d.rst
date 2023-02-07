mindspore.nn.BatchNorm1d
=========================

.. py:class:: mindspore.nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCHW')

    在二维或三维输入（mini-batch 一维输入或二维输入）上应用批归一化（Batch Normalization Layer），避免内部协变量偏移。归一化在卷积网络中被广泛的应用。请见论文 `Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ 。

    使用mini-batch数据和学习参数进行训练，计算公式如下。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. note::
        BatchNorm的实现在图模式和PyNative模式下是不同的，因此不建议在网络初始化后更改其模式。

    参数：
        - **num_features** (int) - 特征数量或输入 `x` 中的通道数量 `C` 。
        - **eps** (float) - :math:`\epsilon` 加在分母上的值，以确保数值稳定。默认值：1e-5。
        - **momentum** (float) - 动态均值和动态方差所使用的动量。默认值：0.9。
        - **affine** (bool) - bool类型。设置为True时，可学习到 :math:`\gamma` 和 :math:`\beta` 值。默认值：True。
        - **gamma_init** (Union[Tensor, str, Initializer, numbers.Number]) - :math:`\gamma` 参数的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'等。默认值：'ones'。
        - **beta_init** (Union[Tensor, str, Initializer, numbers.Number]) - :math:`\beta` 参数的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'等。默认值：'zeros'。
        - **moving_mean_init** (Union[Tensor, str, Initializer, numbers.Number]) - 动态平均值的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'等。默认值：'zeros'。
        - **moving_var_init** (Union[Tensor, str, Initializer, numbers.Number]) - 动态方差的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'等。默认值：'ones'。
        - **use_batch_statistics** (bool) - 如果为True，则使用当前批次数据的平均值和方差值。如果为False，则使用指定的平均值和方差值。如果为None，训练时，将使用当前批次数据的均值和方差，并更新动态均值和方差，验证过程将直接使用动态均值和方差。默认值：None。
        - **data_format** (str) - 数据格式可为'NHWC'或'NCHW'。默认值：'NCHW'。

    输入：
        - **x** (Tensor) - 输入shape为 :math:`(N, C)` 或 :math:`(N, C, L)` 的Tensor，其中 `N` 为batch， `C` 为特征数量或通道数量， `L` 为序列长度。

    输出：
        Tensor，归一化后的Tensor，shape为 :math:`(N, C)` 或 :math:`(N, C, L)` 。

    异常：
        - **TypeError** - `num_features` 不是整数。
        - **TypeError** - `eps` 不是浮点数。
        - **ValueError** - `num_features` 小于1。
        - **ValueError** - `momentum` 不在范围[0, 1]内。
