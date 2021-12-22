mindspore.ops.Sigmoid
=====================

.. py:class:: mindspore.ops.Sigmoid()

    Sigmoid激活函数。

    逐元素计算Sgmoid激活函数。Sigmoid函数定义为：

    .. math::

        \text{sigmoid}(x_i) = \frac{1}{1 + \exp(-x_i)}

    其中， :math:`x_i` 是输入Tensor的一个元素。

    **输入：**

    - **input_x** (Tensor) - 任意维度的Tensor，数据类型为float16或float32。

    **输出：**

    Tensor，数据类型和shape与 `input_x` 的相同。

    **异常：**

    - **TypeError** - `input_x` 的数据类型既不是float16也不是float32。
    - **TypeError** - `input_x` 不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> input_x = Tensor(np.array([1, 2, 3, 4, 5]), mindspore.float32)
    >>> sigmoid = ops.Sigmoid()
    >>> output = sigmoid(input_x)
    >>> print(output)
    [0.7310586  0.880797   0.95257413 0.98201376 0.9933072 ]
    