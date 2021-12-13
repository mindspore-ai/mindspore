mindspore.ops.SeLU
==================

.. py:class:: mindspore.ops.SeLU(*args, **kwargs)

    按元素计算输入Tensor的SeLU（scaled exponential Linear Unit）函数。

    该激活函数定义为：

    .. math::
        E_{i} =
        scale *
        \begin{cases}
        x_{i}, &\text{if } x_{i} \geq 0; \cr
        \text{alpha} * (\exp(x_i) - 1), &\text{otherwise.}
        \end{cases}

    其中， :math:`alpha` 和 :math:`scale` 是预定义的常量（ :math:`alpha=1.67326324` ， :math:`scale=1.05070098` ）。

    更多详细信息，请参见 `Self-Normalizing Neural Networks <https://arxiv.org/abs/1706.02515>`_ 。

    **输入：**

    - **input_x** (Tensor) - shape为 :math:`(N, *)` 的Tensor，其中， :math:`*` 表示任意的附加维度数，数据类型为float16或float32。

    **输出：**

    Tensor，数据类型和shape与 `input_x` 的相同。

    **支持平台：**

    ``Ascend``

    **异常：**

    - **TypeError** - `input_x` 的数据类型既不是float16也不是float32。

    **样例：**

    >>> input_x = Tensor(np.array([[-1.0, 4.0, -8.0], [2.0, -5.0, 9.0]]), mindspore.float32)
    >>> selu = ops.SeLU()
    >>> output = selu(input_x)
    >>> print(output)
    [[-1.1113307 4.202804 -1.7575096]
    [ 2.101402 -1.7462534 9.456309 ]]
