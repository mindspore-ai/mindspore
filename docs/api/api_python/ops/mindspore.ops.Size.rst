mindspore.ops.Size
==================

.. py:class:: mindspore.ops.Size(*args, **kwargs)

    返回Tensor的大小。

    返回一个整数Scalar，表示输入的元素大小，即Tensor中元素的总数。

    **输入：**

    - **input_x** (Tensor) - Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。数据类型为Number。

    **输出：**

    整数，表示 `input_x` 元素大小的Scalar。它的值为 :math:`size=x_1*x_2*...x_R` 。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> input_x = Tensor(np.array([[2, 2], [2, 2]]), mindspore.float32)
    >>> size = ops.Size()
    >>> output = size(input_x)
    >>> print(output)
    4
    