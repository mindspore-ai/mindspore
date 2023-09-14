mindspore.nn.InstanceNorm1d
============================

.. py:class:: mindspore.nn.InstanceNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, gamma_init='ones', beta_init='zeros', dtype=mstype.float32)

    该层在三维输入（带有额外通道维度的mini-batch一维输入）上应用实例归一化。详见论文 `Instance Normalization:
    The Missing Ingredient for Fast Stylization <https://arxiv.org/abs/1607.08022>`_ 。
    使用mini-batch数据和学习参数进行训练，参数见如下公式。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 和 :math:`\beta` 是可学习的参数向量，如果 `affine` 为True，则大小为 `num_features` 。通过偏置估计函数计算标准偏差。

    此层使用从训练和验证模式的输入数据计算得到的实例数据。

    InstanceNorm1d和BatchNorm1d类似。不同之处在于InstanceNorm1d应用于RGB图像等通道数据的每个通道，而BatchNorm1d通常应用于批处理。

    .. note::
        需要注意的是，更新滑动平均和滑动方差的公式为 :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times x_t + \text{momentum} \times \hat{x}` ,其中 :math:`\hat{x}` 是估计的统计量， :math:`x_t` 是新的观察值。

    参数：
        - **num_features** (int) - 通道数量，输入Tensor shape :math:`(N, C, L)` 中的 `C` 。
        - **eps** (float) - 添加到分母中的值，以确保数值稳定。默认值： ``1e-5`` 。
        - **momentum** (float) - 动态均值和动态方差所使用的动量。默认值： ``0.1`` 。
        - **affine** (bool) - bool类型。设置为True时，可以学习gamma和beta参数。默认值： ``True`` 。
        - **gamma_init** (Union[Tensor, str, Initializer, numbers.Number]) - gamma参数的初始化方法。str的值引用自函数 `initializer` ，包括 ``'zeros'`` 、 ``'ones'`` 等。使用Tensor作为初始化参数时，shape必须为 :math:`(C)` 。默认值： ``'ones'`` 。
        - **beta_init** (Union[Tensor, str, Initializer, numbers.Number]) - beta参数的初始化方法。str的值引用自函数 `initializer` ，包括 ``'zeros'`` 、 ``'ones'`` 等。使用Tensor作为初始化参数时，shape必须为 :math:`(C)` 。默认值： ``'zeros'`` 。
        - **dtype** (:class:`mindspore.dtype`) - Parameters的dtype。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C, L)` 的Tensor。数据类型为float16或float32。

    输出：
        Tensor，归一化，缩放，偏移后的Tensor，其shape为 :math:`(N, C, L)` 。类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `num_features` 不是整数。
        - **TypeError** - `eps` 的类型不是float。
        - **TypeError** - `momentum` 的类型不是float。
        - **TypeError** - `affine` 不是bool。
        - **TypeError** - `gamma_init` / `beta_init` 的类型不相同，或者初始化的元素类型不是float32。
        - **ValueError** - `num_features` 小于1。
        - **ValueError** - `momentum` 不在范围[0, 1]内。
        - **ValueError** - `gamma_init` / `beta_init` 的shape不为 :math:`(C)` 。
        - **KeyError** - `gamma_init` / `beta_init` 中的任何一个是str，并且不存在继承自 `Initializer` 的同义类。
