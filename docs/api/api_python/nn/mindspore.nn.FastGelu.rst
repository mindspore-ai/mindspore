mindspore.nn.FastGelu
======================

.. py:class:: mindspore.nn.FastGelu

    快速高斯误差线性单元激活函数（Fast Gaussian Error Linear Units activation function）。

    对输入的每个元素计算FastGelu。

    FastGelu定义如下：

    .. math::
        FastGelu(x_i) = \frac {x_i} {1 + \exp(-1.702 * \left| x_i \right|)} *
                           \exp(0.851 * (x_i - \left| x_i \right|))

    其中 :math:`x_i` 是输入的元素。

    **输入：**

    - **x** (Tensor) - 用于计算FastGelu的Tensor。数据类型为float16或float32。shape为 :math:`(N,*)` ，其中 :math:`*` 表示任意的附加维度数。

    **输出：**

    Tensor，具有与 `x` 相同的数据类型和shape。

    **异常：**

    - **TypeError** - `x` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend``

    **样例：**

    >>> x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    >>> fast_gelu = nn.FastGelu()
    >>> output = fast_gelu(x)
    >>> print(output)
    [[-1.5418735e-01  3.9921875e+00 -9.7473649e-06]
     [ 1.9375000e+00 -1.0052517e-03  8.9824219e+00]]
