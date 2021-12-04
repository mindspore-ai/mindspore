mindspore.nn.ReLU
=================

.. py:class:: mindspore.nn.ReLU

    修正线性单元激活函数（Rectified Linear Unit activation function）。

    按元素返回 :math:`\max(x,\  0)` 。特别说明，负数输出值会被修改为0，正数输出不受影响。

    .. math::

        \text{ReLU}(x) = (x)^+ = \max(0, x),

    ReLU相关图参见 `ReLU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_rectified_linear.svg>`_ 。

    **输入：**

    - **x** (Tensor) - 用于计算ReLU的Tensor。数据类型为Number。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度数。

    **输出：**

    Tensor，具有与 `x` 相同的数据类型和shape。

    **异常：**

    - **TypeError** - `x` 的数据类型不是Number。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor(np.array([-1, 2, -3, 2, -1]), mindspore.float16)
    >>> relu = nn.ReLU()
    >>> output = relu(x)
    >>> print(output)
    [0. 2. 0. 2. 0.]
