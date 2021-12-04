mindspore.ops.GeLU
==================

.. py:class:: mindspore.ops.GeLU(*args, **kwargs)

    高斯误差线性单元激活函数（Gaussian Error Linear Units activation function）。

    `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_ 描述了GeLU函数。
    此外，也可以参考 `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ 。

    GeLU函数定义如下：

    .. math::
        \text{output} = 0.5 * x * (1 + tanh(x / \sqrt{2})),

    其中 :math:`tanh` 是双曲正切函数。

    **输入：**

    - **x** (Tensor) - 用于计算GeLU函数的Tensor，数据类型为float16或float32。

    **输出：**

    Tensor，数据类型和shape与 `x` 的相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。

    **支持平台：**

    ``Ascend`` ``GPU`` ``CPU``

    **样例：**

    >>> x = Tensor(np.array([1.0, 2.0, 3.0]), mindspore.float32)
    >>> gelu = ops.GeLU()
    >>> result = gelu(x)
    >>> print(result)
    [0.841192  1.9545976  2.9963627]
