mindspore.ops.Div
=================

.. py:class:: mindspore.ops.Div(*args, **kwargs)

    按元素计算第一输入Tensor除以第二输入Tensor的商。

    输入 `x` 和 `y` 遵循隐式类型转换规则，使数据类型保持一致。输入必须是两个Tensor，或一个Tensor和一个Scalar。当输入是两个Tensor时，它们的数据类型不能同时是bool，它们的shape可以广播。当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。

    .. math::

        out_{i} = \frac{x_i}{y_i}

    **输入：**

    - **x** (Union[Tensor, Number, bool]) - 第一个输入，是一个Number、bool值或数据类型为Number或bool的Tensor。
    - **y** (Union[Tensor, Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个Number或bool值，或数据类型为Number或bool的Tensor。

    **输出：**

    Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高的类型。

    **异常：**

    - **TypeError** - `x` 和 `y` 都不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> # 用例1：两个输入的数据类型和shape相同
    >>> x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.float32)
    >>> y = Tensor(np.array([3.0, 2.0, 3.0]), mindspore.float32)
    >>> div = ops.Div()
    >>> output = div(x, y)
    >>> print(output)
    [-1.3333334  2.5        2.        ]
    >>> # 用例2：两个输入的数据类型和shape不同
    >>> x = Tensor(np.array([-4.0, 5.0, 6.0]), mindspore.int32)
    >>> y = Tensor(2, mindspore.float32)
    >>> output = div(x, y)
    >>> print(output)
    [-2.  2.5  3.]
    >>> print(output.dtype)
    Float32
    