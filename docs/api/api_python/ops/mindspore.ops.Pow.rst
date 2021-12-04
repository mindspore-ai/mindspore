mindspore.ops.Pow
==================

.. py:class:: mindspore.ops.Pow(*args, **kwargs)

    计算 `x` 中每个元素的 `y` 的幂次。

    输入 `x` 和 `y` 遵循隐式类型转换规则，使数据类型保持一致。
    输入必须是两个Tensor，或一个Tensor和一个Scalar。
    当输入是两个Tensor时，它们的数据类型不能同时是bool，它们的shape可以广播。
    当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。

    .. math::

        out_{i} = x_{i} ^{ y_{i}}

    **输入：**

    - **x** (Union[Tensor, Number, bool]) - 第一个输入，是一个Number、bool值或数据类型为Number或bool的Tensor。
    - **y** (Union[Tensor, Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个Number或bool值，或数据类型为Number或bool的Tensor。

    **输出：**
    
    Tensor，shape与广播后的shape相同，数据类型为两个输入中精度较高的类型。

    **异常：**

    - **TypeError** - `x` 和 `y` 不是Tensor、Number或bool。
    - **ValueError** - `x` 和 `y` 的shape不相同。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
    >>> y = 3.0
    >>> pow = ops.Pow()
    >>> output = pow(x, y)
    >>> print(output)
    [ 1.  8. 64.]
    >>>
    >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
    >>> y = Tensor(np.array([2.0, 4.0, 3.0]), mindspore.float32)
    >>> pow = ops.Pow()
    >>> output = pow(x, y)
    >>> print(output)
    [ 1. 16. 64.]
    