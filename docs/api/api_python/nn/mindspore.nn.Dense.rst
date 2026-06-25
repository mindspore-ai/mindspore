mindspore.nn.Dense
===================

.. py:class:: mindspore.nn.Dense(in_channels, out_channels, weight_init=None, bias_init=None, has_bias=True, activation=None, dtype=mstype.float32)

    全连接层，对输入执行密集连接操作。

    公式如下：

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias}),

    其中 :math:`X` 是输入Tensor， :math:`\text{activation}` 是激活函数， :math:`\text{kernel}` 是一个权重矩阵，其数据类型与 :math:`X` 相同， :math:`\text{bias}` 是一个偏置向量，其数据类型与 :math:`X` 相同（仅当 `has_bias` 为 ``True`` 时）。

    .. warning::
        - 在Ascend硬件平台下，设置PYNATIVE或KBK模式时，如果 `bias` 为 ``False``， `x` 不可以大于6D。
        - 从2.9.0（不含）之后版本开始，该接口将重命名为 `Linear`。

    参数：
        - **in_channels** (int) - Dense层输入Tensor的空间维度。
        - **out_channels** (int) - Dense层输出Tensor的空间维度。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 权重参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 :func:`mindspore.common.initializer.initializer` 。默认值： ``None`` ，权重使用HeUniform初始化。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 偏置参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 :func:`mindspore.common.initializer.initializer` 。默认值： ``None`` ，偏差使用Uniform初始化。
        - **has_bias** (bool，可选) - 是否使用偏置向量 :math:`\text{bias}` 。默认值： ``True`` 。
        - **activation** (Union[str, Cell, Primitive, None]，可选) - 应用于全连接层输出的激活函数。可指定激活函数名，如'relu'，或具体激活函数，如 :class:`mindspore.nn.ReLU` 。默认值： ``None`` 。
        - **dtype** (:class:`mindspore.dtype`，可选) - Parameter的数据类型。默认值： ``mstype.float32`` 。
          当 `weight_init` 是Tensor时，Parameter的数据类型与 `weight_init` 的数据类型一致，其他情况Parameter的数据类型跟 `dtype` 一致， `bias_init` 同理。

    输入：
        - **x** (Tensor) - shape为 :math:`(*, in\_channels)` 的Tensor。参数中的 `in_channels` 应等于输入中的 :math:`in\_channels` 。

    输出：
        shape为 :math:`(*, out\_channels)` 的Tensor。

    异常：
        - **TypeError** - `in_channels` 或 `out_channels` 不是整数。
        - **TypeError** - `has_bias` 不是bool值。
        - **TypeError** - `activation` 不是str、Cell、Primitive或者None。
        - **ValueError** - `weight_init` 的shape长度不等于2，`weight_init` 的shape[0]不等于 `out_channels`，或者 `weight_init` 的shape[1]不等于 `in_channels`。
        - **ValueError** - `bias_init` 的shape长度不等于1或 `bias_init` 的shape[0]不等于 `out_channels`。
        - **RuntimeError** - 在Ascend硬件平台下，设置PYNATIVE或KBK模式时， `bias` 为 ``False`` 且 `x` 大于6D。
