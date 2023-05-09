mindspore.ops.rrelu
===================

.. py:function:: mindspore.ops.rrelu(input, lower=1.0 / 8, upper=1.0 / 3)

    Randomized Leaky ReLU激活函数。

    该激活函数定义如下：

    .. math::
        \text{rrelu}(input_{ji}) = \begin{cases}input_{ji}, &\text{if } input_{ji} \geq 0; \cr
        {\alpha_{ji}} * input_{ji}, &\text{otherwise.}\end{cases}

    其中，:math:`\alpha_{ji}` ~ :math:`U(l, u)`, :math:`l \le u`.

    更多细节详见 `Empirical Evaluation of Rectified Activations in Convolution Network <https://arxiv.org/pdf/1505.00853.pdf>`_。

    参数：
        - **input** (Tensor) - 计算RReLU的任意维度的Tensor。
        - **lower** (Union[int, float]) - x<0时激活函数的斜率的下界，默认值： ``1.0 / 8`` 。
        - **upper** (Union[int, float]) - x<0时激活函数的斜率的上界，默认值： ``1.0 / 3`` 。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `lower` 不是浮点数或整数。
        - **TypeError** - `upper` 不是浮点数或整数。
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 内的数据类型不是mindspore.float16或mindspore.float32。
        - **ValueError** - `lower` 大于 `upper`。
