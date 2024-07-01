mindspore.mint.nn.Linear
==========================

.. py:class:: mindspore.mint.nn.Linear(in_features, out_features, bias=True, weight_init=None, bias_init=None, dtype=None)

    全连接层。

    适用于输入的密集连接层。公式如下：

    .. math::
        \text{outputs} = X * kernel + bias

    其中  :math:`X` 是输入Tensor， :math:`\text{kernel}` 是一个权重矩阵，其数据类型与 :math:`X` 相同， :math:`\text{bias}` 是一个偏置向量，其数据类型与 :math:`X` 相同（仅当has_bias为True时）。

    参数：
        - **in_features** (int) - Linear层输入Tensor的空间维度。
        - **out_features** (int) - Linear层输出Tensor的空间维度。
        - **bias** (bool) - 是否使用偏置向量 :math:`\text{bias}` 。默认值： ``True`` 。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]) - 权重参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 `initializer`。默认值： ``None`` ，权重使用HeUniform初始化。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]) - 偏置参数的初始化方法。数据类型与 `x` 相同。str的值引用自函数 `initializer`。默认值： ``None`` ，偏差使用Uniform初始化。
        - **dtype** (:class:`mindspore.dtype`) - Parameter的数据类型。默认值： ``None`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(*, in\_features)` 的Tensor。参数中的 `in_features` 应等于输入中的 :math:`in\_features` 。

    输出：
        shape为 :math:`(*, out\_features)` 的Tensor。

    异常：
        - **TypeError** - `in_features` 或 `out_features` 不是整数。
        - **TypeError** - `bias` 不是bool值。
        - **ValueError** - `weight_init` 的shape长度不等于2，`weight_init` 的shape[0]不等于 `out_features`，或者 `weight_init` 的shape[1]不等于 `in_features`。
        - **ValueError** - `bias_init` 的shape长度不等于1或 `bias_init` 的shape[0]不等于 `out_features`。
