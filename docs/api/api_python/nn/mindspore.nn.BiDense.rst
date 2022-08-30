mindspore.nn.BiDense
====================

.. py:class:: mindspore.nn.BiDense(in1_channels, in2_channels, out_channels, weight_init=None, bias_init=None, has_bias=True)

    双线性全连接层。

    两个输入的密集连接层。公式如下：

    .. math::
        y = x_1^T A x_2 + b,

    其中  :math:`x_{1}` 是第一个输入Tensor，:math:`x_{2}` 是第二个输入Tensor，:math:`A` 是一个权重矩阵，其数据类型与 :math:`x_{*}` 相同， :math:`b` 是一个偏置向量，其数据类型与 :math:`x_{*}` 相同（仅当has_bias为True时）。

    参数：
        - **in1_channels** (int) - BiDense层第一个输入Tensor的空间维度。
        - **in2_channels** (int) - BiDense层第二个输入Tensor的空间维度。
        - **out_channels** (int) - BiDense层输出Tensor的空间维度。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。str的值引用自函数 `initializer`。默认值：None。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。str的值引用自函数 `initializer`。默认值：None。
        - **has_bias** (bool) - 是否使用偏置向量 :math:`\text{bias}` 。默认值：True。

    形状：
        - **input1** - :math:`(*, H_{in1})`，其中 :math:`H_{in1}=\text{in1_channels}`，
          :math:`*` 为任意维度. input1除最后一维外的维度需与其他输入保持一致。
        - **input2** - :math:`(*, H_{in2})`，其中 :math:`H_{in2}=\text{in2_channels}`，
          :math:`*` 为任意维度. input2除最后一维外的维度需与其他输入保持一致。
        - **output** - :math:`(*, H_{out})`，其中 :math:`H_{out}=\text{out_channels}`，
          :math:`*` 为任意维度. output除最后一维外的维度需与所有输入保持一致。

    数据类型：
        - **input1** (Tensor) - 数据类型必须为float16或float32，且与input2一致。
        - **input2** (Tensor) - 数据类型必须为float16或float32，且与input1一致。
        - **output** (Tensor) - 数据类型与输入保持一致。

    权重：
        - **weight** (Parameter) - 权重参数，shape为 :math:`(\text{out_channels}, \text{in1_channels}, \text{in2_channels})`。
          当 `weight_init` 设为 `None` 时，其初始化值服从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`，其中 :math:`k = \frac{1}{\text{in1_channels}}`。
        - **bias** (Parameter) - 偏置参数，shape为 :math:`(\text{out_channels})`。
          当 `has_bias` 设为 `True` 且 `bias_init` 设为 `None` 时，其初始化值服从 :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`，其中 :math:`k = \frac{1}{\text{in1_channels}}`。

    异常：
        - **TypeError** - `in_channels` 或  `out_channels` 不是整数。
        - **TypeError** - `has_bias` 不是bool值。
        - **ValueError** - `weight_init` 的shape长度不等于3，`weight_init` 的shape[0]不等于 `out_channels`，或者 `weight_init` 的shape[1]不等于 `in1_channels`，或者 `weight_init` 的shape[2]不等于 `in2_channels`。
        - **ValueError** - `bias_init` 的shape长度不等于1或 `bias_init` 的shape[0]不等于 `out_channels`。