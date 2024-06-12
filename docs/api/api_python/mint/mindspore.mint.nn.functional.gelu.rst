mindspore.mint.nn.functional.gelu
====================================

.. py:function:: mindspore.mint.nn.functional.gelu(input, approximate='none')

    高斯误差线性单元激活函数。

    GeLU的描述可以在 `Gaussian Error Linear Units (GELUs) <https://arxiv.org/abs/1606.08415>`_ 这篇文章中找到。
    也可以去查询 `BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding <https://arxiv.org/abs/1810.04805>`_ 。

    当 `approximate` 为 `none` ，GELU的定义如下：

    .. math::
        GELU(x_i) = x_i*P(X < x_i),

    其中 :math:`P` 是标准高斯分布的累积分布函数， :math:`x_i` 是输入的元素。

    当 `approximate` 为 `tanh` ，GELU的定义如下：

    .. math::
        GELU(x_i) = 0.5 * x_i * (1 + \tanh(\sqrt(2 / \pi) * (x_i + 0.044715 * x_i^3)))

    GELU函数图：

    .. image:: ../images/GELU.png
        :align: center

    参数：
        - **input** (Tensor) - 用于计算GELU的Tensor。数据类型是float16、float32或float64。
        - **approximate** (str) - gelu近似算法。有两种：``'none'`` 和 ``'tanh'`` 。默认值： ``'none'`` 。

    返回：
        Tensor，具有与 `input` 相同的数据类型和shape。

    异常：
        - **TypeError** - 如果 `input` 的数据类型不是Tensor。
        - **TypeError** - `input` 的数据类型既不是bfloat16、float16、float32或者float64。
        - **ValueError** - 如果 `approximate` 的值既不是 `none` 也不是 `tanh`。
