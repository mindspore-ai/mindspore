mindspore.ops.Dense
===================

.. py:class:: mindspore.ops.Dense(in_channels, out_channels, weight_init='normal', bias_init='zeros', has_bias=True, activation=None)

    全连接融合算子。

    适用于输入的密集连接算子。算子的实现如下：

    .. math::
        \text{output} = \text{x} * \text{w} + \text{b},

    其中  :math:`x` 是输入Tensor， :math:`\text{w}` 是一个权重矩阵，其数据类型与 :math:`x` 相同， :math:`\text{b}` 是一个偏置向量，其数据类型与 :math:`b` 相同（仅当has_bias为True时）。

    参数：
        - **has_bias** (bool) - 是否使用偏置向量 :math:`\text{bias}` 。默认值： ``True`` 。

    输入：
        - **x** (Union[Tensor, Parameter]) - 输入Tensor，其数据类型为float16、float32或float64。
        - **w** (Union[Tensor, Parameter]) - 权重Tensor，其数据类型为float16、float32或float64。
        - **b** (Union[Tensor, Parameter]) - 偏置Tensor，其数据类型为float16、float32或float64。

    输出：
        shape为 :math:`(*x.shape[:-1], w.shape[0])` 的Tensor。

    异常：
        - **TypeError** - `has_bias` 不是bool值。
