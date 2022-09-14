mindspore.nn.BatchNorm3d
=========================

.. py:class:: mindspore.nn.BatchNorm3d(num_features, eps=1e-5, momentum=0.9, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros', moving_var_init='ones', use_batch_statistics=None, data_format='NCDHW')

    对输入的五维数据进行批归一化(Batch Normalization Layer)。

    在五维输入（带有附加通道维度的mini-batch 三维输入）上应用批归一化，避免内部协变量偏移。归一化在卷积网络中得到了广泛的应用。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. note::
        BatchNorm的实现在图模式和PyNative模式下是不同的，因此不建议在网络初始化后更改其模式。

        需要注意的是，更新running_mean和running_var的公式为 :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times x_t + \text{momentum} \times \hat{x}` ,其中 :math:`\hat{x}` 是估计的统计量， :math:`x_t` 是新的观察值。

    参数：
        - **num_features** (int) - 指定输入Tensor的通道数量。输入Tensor的size为(N, C, D, H, W)。
        - **eps** (float) - 加在分母上的值，以确保数值稳定。默认值：1e-5。
        - **momentum** (float) - 动态均值和动态方差所使用的动量。默认值：0.9。
        - **affine** (bool) - bool类型。设置为True时，可以学习gama和beta。默认值：True。
        - **gamma_init** (Union[Tensor, str, Initializer, numbers.Number]) - gamma参数的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'等。默认值：'ones'。
        - **beta_init** (Union[Tensor, str, Initializer, numbers.Number]) - beta参数的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'等。默认值：'zeros'。
        - **moving_mean_init** (Union[Tensor, str, Initializer, numbers.Number]) - 动态均值和动态方差所使用的动量。平均值的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'等。默认值：'zeros'。
        - **moving_var_init** (Union[Tensor, str, Initializer, numbers.Number]) - 动态均值和动态方差所使用的动量。方差的初始化方法。str的值引用自函数 `mindspore.common.initializer` ，包括'zeros'、'ones'等。默认值：'ones'。
        - **use_batch_statistics** (bool) - 如果为True，则使用当前批次数据的平均值和方差值。如果为False，则使用指定的平均值和方差值。如果为None，训练时，将使用当前批次数据的均值和方差，并更新动态均值和方差，验证过程将直接使用动态均值和方差。默认值：None。

    输入：
        - **x** (Tensor) - 输入shape为 :math:`(N, C_{in}, D_{in}, H_{in}, W_{in})` 的Tensor。

    输出：
        Tensor，归一化后的Tensor，shape为 :math:`(N, C_{out}, D_{out},H_{out}, W_{out})`。

    异常：
        - **TypeError** - `num_features` 不是整数。
        - **TypeError** - `eps` 不是浮点数。
        - **ValueError** - `num_features` 小于1。
        - **ValueError** - `momentum` 不在范围[0, 1]内。
        - **ValueError** - `data_format` 不是'NCDHW'。
