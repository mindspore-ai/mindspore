mindspore.nn.GELU
==================

.. py:class:: mindspore.nn.GELU(approximate=True)

    高斯误差线性单元激活函数（Gaussian error linear unit activation function）。

    .. warning::
        从2.9.0（不含）之后版本开始，`approximate` 将从布尔值改为字符串，默认值也将调整。

    对输入的每个元素计算GELU，输入可以是任意有效shape的Tensor。

    GELU的定义如下：

    .. math::
        GELU(x_i) = x_i*P(X < x_i),

    其中 :math:`P` 是标准高斯分布的累积分布函数， :math:`x_i` 是输入的元素。

    GELU函数图：

    .. image:: ../images/GELU.png
        :align: center

    参数：
        - **approximate** (bool，可选) - 是否启用approximation。默认值： ``True`` 。如果approximate的值为 ``True`` ，则高斯误差线性激活函数为：

          :math:`0.5 * x * (1 + tanh(\sqrt(2 / \pi) * (x + 0.044715 * x^3)))` 。

          否则为： :math:`x * P(X <= x) = 0.5 * x * (1 + erf(x / \sqrt(2)))`，其中 :math:`P(X) ~ N(0, 1)` 。

    .. note::
        - 在计算gelu的输入梯度时，当输入为inf，Ascend与GPU在反向传播输出之间存在差异。
        - 当输入x为-inf时，Ascend的计算结果为0，GPU的计算结果为nan。
        - 当输入x为inf时，Ascend的计算结果为梯度dy，GPU的计算结果为nan。
        - 数学意义上，Ascend的计算结果精度更高。

    输入：
        - **x** (Tensor) - 用于计算GELU的Tensor。数据类型为float16、float32或float64。shape是 :math:`(N,*)` ， :math:`*` 表示任意的附加维度数。

    输出：
        Tensor，具有与 `x` 相同的数据类型和shape。

    异常：
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。
