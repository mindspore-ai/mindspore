mindspore.ops.GeLU
==================

.. py:class:: mindspore.ops.GeLU

    高斯误差线性单元激活函数（Gaussian Error Linear Units activation function）。

    在 `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_ 文章中对GeLU函数进行了介绍。
    此外，也可以参考 `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ 。

    GeLU函数定义如下：

    .. math::
        GELU(x_i) = x_i*P(X < x_i)

    其中 :math:`P` 是标准高斯分布的累积分布函数， :math:`x_i` 是输入的元素。

    .. note::
        在计算gelu的输入梯度时，当输入为inf，Ascend与GPU在反向传播输出之间存在差异。
        当输入x为-inf时，Ascend的计算结果为0，GPU的计算结果为nan。
        当输入x为inf时，Ascend的计算结果为梯度dy，GPU的计算结果为nan。
        数学意义上，Ascend的计算结果精度更高。

    输入：
        - **x** (Tensor) - 激活函数GeLU的输入，数据类型为float16、float32或float64。

    输出：
        Tensor，数据类型和shape与 `x` 的相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。
