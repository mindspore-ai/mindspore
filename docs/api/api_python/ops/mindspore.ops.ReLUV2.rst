mindspore.ops.ReLUV2
====================

.. py:class:: mindspore.ops.ReLUV2(*args, **kwargs)

    线性修正单元激活函数（Rectified Linear Unit activation function）。

    按元素返回 :math:`\max(x,\  0)` 。特别说明，负数输出值会被修改为0，正数输出不受影响。

    .. math::

        \text{ReLU}(x) = (x)^+ = \max(0, x)，

    .. note::

        与 `ReLu` 的区别在于该算子多输出一个mask，且算子的kernel与 `ReLu` 的不同。

    **输入：**

    - **input_x** (Tensor) - 输入Tensor必须是4-D Tensor。

    **输出：**

    - **output** (Tensor) - 数据类型和shape与 `input_x` 的相同。
    - **mask** (Tensor) - 数据类型必须为uint8的Tensor。

    **异常：**

    - **TypeError** - `input_x` 不是Tensor。
    - **ValueError** - `input_x` 的shape不是4-D。

    **支持平台：**

    ``Ascend``

    **样例：**

    >>> input_x = Tensor(np.array([[[[1, -2], [-3, 4]], [[-5, 6], [7, -8]]]]), mindspore.float32)
    >>> relu_v2 = ops.ReLUV2()
    >>> output, mask= relu_v2(input_x)
    >>> print(output)
    [[[[1. 0.]
       [0. 4.]]
       [[0. 6.]
       [7. 0.]]]]
    >>> print(mask)
    [[[[[1 0]
        [2 0]]
        [[2 0]
        [1 0]]]]]
