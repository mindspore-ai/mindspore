mindspore.nn.GELU
==================

.. py:class:: mindspore.nn.GELU

    高斯误差线性单元激活函数（Gaussian error linear unit activation function）。

    对输入的每个元素计算GELU。

    GELU的定义如下：

    .. math::
        GELU(x_i) = x_i*P(X < x_i),


    其中 :math:`P` 是标准高斯分布的累积分布函数， :math:`x_i` 是输入的元素。

    GELU相关图参见 `GELU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_gelu.png>`_  。

    **输入：**

    - **x** （Tensor） - 用于计算GELU的Tensor。数据类型为float16或float32。shape是 :math:`(N,*)` ， :math:`*` 表示任意的附加维度数。

    **输出：**

    Tensor，具有与 `x` 相同的数据类型和shape。

    **异常：**

    - **TypeError** - `x` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    >>> gelu = nn.GELU()
    >>> output = gelu(x)
    >>> print(output)
    [[-1.5880802e-01  3.9999299e+00 -3.1077917e-21]
    [ 1.9545976e+00 -2.2918017e-07  9.0000000e+00]]