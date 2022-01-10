mindspore.ops.Log
=================

.. py:class:: mindspore.ops.Log()

    逐元素返回Tensor的自然对数。

    .. math::
        y_i = log_e(x_i)

    .. warning::

        如果算子Log的输入值在(0，0.01]或[0.95，1.05]范围内，则输出精度可能会存在误差。

    .. note::
        Ascend上输入Tensor的维度要小于等于8，CPU上输入Tensor的维度要小于8。

    **输入：**

    - **x** (Tensor) - 任意维度的输入Tensor。该值必须大于0。

    **输出：**

    Tensor，具有与 `x` 相同的shape。

    **异常：**

    - **TypeError** - `x` 不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**


    >>> x = Tensor(np.array([1.0, 2.0, 4.0]), mindspore.float32)
    >>> log = ops.Log()
    >>> output = log(x)
    >>> print(output)
    [0.        0.6931472 1.3862944]
