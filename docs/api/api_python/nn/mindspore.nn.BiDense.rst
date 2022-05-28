mindspore.nn.BiDense
====================

.. py:class:: mindspore.nn.BiDense(in1_channels, in2_channels, out_channels, weight_init=None, bias_init=None, has_bias=True)

    双线性全连接层。

    公式如下：

    .. math::
        \text{outputs} = \text{X_{1}}^{T} * \text{kernel} * \text{X_{2}} + \text{bias},

    其中  :math:`X_{1}` 是第一个输入Tensor，:math:`X_{2}` 是第二个输入Tensor，:math:`\text{kernel}` 是一个权重矩阵，其数据类型与 :math:`X` 相同， :math:`\text{bias}` 是一个偏置向量，其数据类型与 :math:`X` 相同（仅当has_bias为True时）。

    **参数：**

    - **in1_channels** (int) - BiDense层第一个输入Tensor的空间维度。
    - **in2_channels** (int) - BiDense层第二个输入Tensor的空间维度。
    - **out_channels** (int) - BiDense层输出Tensor的空间维度。
    - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 `initializer`。默认值：None。
    - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 `initializer`。默认值：None。
    - **has_bias** (bool) - 是否使用偏置向量 :math:`\text{bias}` 。默认值：True。

    **输入：**

    - **input1** (Tensor) - shape为 :math:`(*,in1\_channels)` 的Tensor。 参数中的 `in1_channels` 应等于输入中的 :math:`in1\_channels` 。
    - **input2** (Tensor) - shape为 :math:`(*,in2\_channels)` 的Tensor。 参数中的 `in2_channels` 应等于输入中的 :math:`in2\_channels` 。

    **输出：**

    shape为 :math:`(*,out\_channels)` 的Tensor。

    **异常：**

    - **TypeError** - `in_channels` 或  `out_channels` 不是整数。
    - **TypeError** - `has_bias` 不是bool值。
    - **ValueError** - `weight_init` 的shape长度不等于3，`weight_init` 的shape[0]不等于 `out_channels`，或者 `weight_init` 的shape[1]不等于 `in1_channels`，或者 `weight_init` 的shape[2]不等于 `in2_channels`。
    - **ValueError** - `bias_init` 的shape长度不等于1或 `bias_init` 的shape[0]不等于 `out_channels`。