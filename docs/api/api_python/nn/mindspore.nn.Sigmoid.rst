mindspore.nn.Sigmoid
=============================

.. py:class:: mindspore.nn.Sigmoid

    Sigmoid激活函数。

    按元素计算Sigmoid激活函数。

    Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)},

    其中 :math:`x_i` 是输入的元素。

    关于Sigmoid的图例见`Sigmoid <https://en.wikipedia.org/wiki/Sigmoid_function#/media/File:Logistic-curve.svg>`_。

    **输入：**

    - **x** (Tensor) - 数据类型为float16或float32的Sgmoid输入。任意维度的Tensor。

    **输出：**

    Tensor，数据类型和shape与 `x` 的相同。

    **异常：**

    **TypeError** - `x` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float16)
    >>> sigmoid = nn.Sigmoid()
    >>> output = sigmoid(x)
    >>> print(output)
    [0.2688  0.11914 0.5     0.881   0.7305 ]
    