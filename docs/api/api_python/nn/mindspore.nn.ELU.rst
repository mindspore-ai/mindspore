mindspore.nn.ELU
=================

.. py:class:: mindspore.nn.ELU(alpha=1.0)

    指数线性单元激活函数（Exponential Linear Uint activation function）。

    对输入的每个元素计算ELU。该激活函数定义如下：

    .. math::
            E_{i} =
            \begin{cases}
            x, &\text{if } x \geq 0; \cr
            \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
            \end{cases}


    ELU相关图参见 `ELU <https://en.wikipedia.org/wiki/Activation_function#/media/File:Activation_elu.svg>`_  。

    **参数：**

    - **alpha** (`float`) – ELU的alpha值，数据类型为浮点数。默认值：1.0。

    **输入：**

    - **x** （Tensor） - 用于计算ELU的Tensor，数据类型为float16或float32。shape为 :math:`(N,*)` ， :math:`*` 表示任意的附加维度数。

    **输出：**

    Tensor，具有与 `x` 相同的数据类型和shape。

    **异常：**

    - **TypeError** - `alpha` 不是浮点数。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。
    - **ValueError** - `alpha` 不等于1.0。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例** :

    >>> x = Tensor(np.array([-1, -2, 0, 2, 1]), mindspore.float32)
    >>> elu = nn.ELU()
    >>> result = elu(x)
    >>> print(result)
    [-0.63212055  -0.86466473  0.  2.  1.]