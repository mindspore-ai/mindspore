mindspore.mint.nn.Linear
==========================

.. py:class:: mindspore.mint.nn.Linear(in_features, out_features, bias=True, weight_init=None, bias_init=None, dtype=None)

    全连接层。

    适用于输入的密集连接层。公式如下：

    .. math::
        \text{outputs} = X * kernel + bias

    其中  :math:`X` 是输入Tensor， :math:`\text{kernel}` 是一个权重矩阵，其数据类型与 :math:`X` 相同， :math:`\text{bias}` 是一个偏置向量，其数据类型与 :math:`X` 相同（仅当参数 `bias` 为True时）。

    .. warning::
        在Ascend硬件平台下，设置PYNATIVE或KBK模式时，如果 `bias` 为 ``False``， `x` 不可以大于6D。

    参数：
        - **in_features** (int) - 输入tensor的空间维度。
        - **out_features** (int) - 输出tensor的空间维度。
        - **bias** (bool，可选) - 是否使用偏置向量 :math:`\text{bias}` 。默认 ``True`` 。
        - **weight_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 权重参数的初始化方法。str的值引用自函数 `initializer`。默认 ``None`` ，权重使用HeUniform初始化。
        - **bias_init** (Union[Tensor, str, Initializer, numbers.Number]，可选) - 偏置参数的初始化方法。str的值引用自函数 `initializer`。默认 ``None`` ，偏置使用Uniform初始化。
        - **dtype** (:class:`mindspore.dtype`，可选) - 参数的数据类型。默认 ``None`` 。

    输入：
        - **x** (Tensor) - shape为 :math:`(*, in\_features)` 的输入tensor。

    输出：
        shape为 :math:`(*, out\_features)` 的tensor。

