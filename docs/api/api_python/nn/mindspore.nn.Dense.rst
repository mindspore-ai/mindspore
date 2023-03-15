mindspore.nn.Dense
===================

.. py:class:: mindspore.nn.Dense(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=True, activation=None)

    全连接层。

    适用于输入的密集连接层。公式如下：

    .. math::
        \text{outputs} = \text{activation}(\text{X} * \text{kernel} + \text{bias}),

    其中  :math:`X` 是输入Tensor， :math:`\text{activation}` 是激活函数， :math:`\text{kernel}` 是一个权重矩阵，其数据类型与 :math:`X` 相同， :math:`\text{bias}` 是一个偏置向量，其数据类型与 :math:`X` 相同（仅当has_bias为True时）。

    参数：
        - **in_channels** (int) - Dense层输入Tensor的空间维度。
        - **out_channels** (int) - Dense层输出Tensor的空间维度。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 `initializer`。默认值：'normal'。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 `initializer`。默认值：'zeros'。
        - **has_bias** (bool) - 是否使用偏置向量 :math:`\text{bias}` 。默认值：True。
        - **activation** (Union[str, Cell, Primitive, None]) - 应用于全连接层输出的激活函数。可指定激活函数名，如'relu'，或具体激活函数，如mindspore.nn.ReLU()。默认值：None。

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
