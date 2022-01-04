mindspore.ops.L2Loss
====================

.. py:class:: mindspore.ops.L2Loss(*args, **kwargs)

    计算Tensor的L2范数的一半，不对结果进行开方。

    把 `input_x` 设为x，输出设为loss。

    .. math::
        loss = sum(x ** 2) / 2

    **输入：**

    - **input_x** (Tensor) - 用于计算L2范数的Tensor。数据类型必须为float16或float32。

    **输出：**

    Tensor，具有与 `input_x` 相同的数据类型。输出Tensor是loss的值，是一个scalar Tensor。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。
    - **TypeError** - `input_x` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例:**

    >>> input_x = Tensor(np.array([1, 2, 3]), mindspore.float16)
    >>> l2_loss = ops.L2Loss()
    >>> output = l2_loss(input_x)
    >>> print(output)
    7.0
