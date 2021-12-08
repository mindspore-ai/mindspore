mindspore.nn.LogSigmoid
=============================

.. py:class:: mindspore.nn.LogSigmoid

    Log Sigmoid激活函数。

    按元素计算Log Sigmoid激活函数。

    Log Sigmoid定义为：

    .. math::
        \text{logsigmoid}(x_{i}) = log(\frac{1}{1 + \exp(-x_i)}),

    其中，:math:`x_i` 是输入Tensor的一个元素。

    **输入：**

    - **x** (Tensor) - Log Sigmoid的输入，数据类型为float16或float32。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度。

    **输出：**

    Tensor，数据类型和shape与 `x` 的相同。

    **异常：**

    **TypeError** - `x` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> net = nn.LogSigmoid()
    >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    >>> output = net(x)
    >>> print(output)
    [-0.31326166 -0.12692806 -0.04858734]
    