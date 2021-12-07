mindspore.ops.Greater
=====================

.. py:class:: mindspore.ops.Greater(*args, **kwargs)

    按元素计算 :math:`x > y` 的bool值。

    输入 `x` 和 `y` 遵循隐式类型转换规则，使数据类型保持一致。
    输入必须是两个Tensor，或一个Tensor和一个Scalar。
    当输入是两个Tensor时，它们的数据类型不能同时是bool，它们的shape可以广播。
    当输入是一个Tensor和一个Scalar时，Scalar只能是一个常数。

    .. math::
        out_{i} =\begin{cases}
            & \text{True,    if } x_{i}>y_{i} \\
            & \text{False,   if } x_{i}<=y_{i}
            \end{cases}

    .. note::
        支持广播。

    **输入：**

    - **x** (Union[Tensor, Number, bool]) - 第一个输入，是一个Number、bool值或数据类型为Number或bool的Tensor。
    - **y** (Union[Tensor, Number, bool]) - 第二个输入，当第一个输入是Tensor时，第二个输入应该是一个Number或bool值，或数据类型为Number或bool的Tensor。

    **输出：**

    Tensor，shape与广播后的shape相同，数据类型为bool。

    **异常：**

    **TypeError** - `x` 和 `y` 都不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor(np.array([1, 2, 3]), mindspore.int32)
    >>> y = Tensor(np.array([1, 1, 4]), mindspore.int32)
    >>> greater = ops.Greater()
    >>> output = greater(x, y)
    >>> print(output)
    [False  True False]
    