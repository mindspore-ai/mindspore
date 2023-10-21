mindspore.nn.LayerNorm
=======================

.. py:class:: mindspore.nn.LayerNorm(normalized_shape, begin_norm_axis=-1, begin_params_axis=-1, gamma_init='ones', beta_init='zeros', epsilon=1e-7, dtype=mstype.float32)

    在mini-batch输入上应用层归一化（Layer Normalization）。

    层归一化在递归神经网络中被广泛的应用。适用单个训练用例的mini-batch输入上应用归一化，详见论文 `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_ 。

    与批归一化（Batch Normalization）不同，层归一化在训练和测试时执行完全相同的计算。
    应用于所有通道和像素，即使batch_size=1也适用。其中 :math:`\gamma` 是通过训练学习出的scale值，:math:`\beta` 是通过训练学习出的shift值。公式如下：

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    参数：
        - **normalized_shape** (Union(tuple[int], list[int])) - 沿轴 `begin_norm_axis ... R - 1` 执行归一化。其中R为输入 `x` 的维度大小。
        - **begin_norm_axis** (int) - 归一化开始计算的轴，取值范围是[-1, R)。默认值： ``-1`` 。
        - **begin_params_axis** (int) - 指定输入参数 :math:`(\gamma, \beta)` 需进行层归一化的开始轴，取值范围是[-1, R)。默认值： ``-1`` 。
        - **gamma_init** (Union[Tensor, str, Initializer, numbers.Number]) - :math:`\gamma` 参数的初始化方法。str的值引用自函数 `initializer` ，包括 ``'zeros'`` 、 ``'ones'`` 、 ``'xavier_uniform'`` 、 ``'he_uniform'`` 等。默认值： ``'ones'`` 。
        - **beta_init** (Union[Tensor, str, Initializer, numbers.Number]) - :math:`\beta` 参数的初始化方法。str的值引用自函数 `initializer` ，包括 ``'zeros'`` 、 ``'ones'`` 、 ``'xavier_uniform'`` 、 ``'he_uniform'`` 等。默认值： ``'zeros'`` 。
        - **epsilon** (float) - 添加到分母中的值（:math:`\epsilon`），以确保数值稳定。默认值： ``1e-7`` 。
        - **dtype** (:class:`mindspore.dtype`) - Parameters的dtype。默认值： ``mstype.float32`` 。

    输入：
        - **x** (Tensor) - `x` 的shape为 :math:`(x_1, x_2, ..., x_R)` ， `input_shape[begin_norm_axis:]` 等于 `normalized_shape` 。

    输出：
        Tensor，归一化后的Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `normalized_shape` 既不是list也不是tuple。
        - **TypeError** - `begin_norm_axis` 或 `begin_params_axis` 不是int。
        - **TypeError** - `epsilon` 不是float。