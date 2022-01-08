mindspore.nn.HShrink
=============================

.. py:class:: mindspore.nn.HShrink(lambd=0.5)

    Hard Shrink激活函数，按输入元素计算输出，公式定义如下：

    .. math::
        \text{HardShrink}(x) =
        \begin{cases}
        x, & \text{ if } x > \lambda \\
        x, & \text{ if } x < -\lambda \\
        0, & \text{ otherwise }
        \end{cases}

    **参数：**

    **lambd** (float) - Hard Shrink公式定义的阈值 :math:`\lambda` 。默认值：0.5。

    **输入：**
        
    - **input_x** (Tensor) - Hard Shrink的输入，数据类型为float16或float32。

    **输出：**

    Tensor，shape和数据类型与输入相同。

    **支持平台：**

    ``Ascend``

    **异常：**

    - **TypeError** - `lambd` 数据类型不是float。
    - **TypeError** - `input_x` 数据类型不是float。

    **样例：**

    >>> input_x = Tensor(np.array([[ 0.5,  1,  2.0],[0.0533,0.0776,-2.1233]]),mstype.float32)
    >>> hshrink = nn.HShrink()
    >>> output = hshrink(input_x)
    >>> print(output)
    [[ 0.      1.      2.    ]
    [ 0.      0.     -2.1233]]
    