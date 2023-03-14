mindspore.nn.RReLU
==================

.. py:class:: mindspore.nn.RReLU(lower=1 / 8, upper=1 / 3)

   Randomized Leaky ReLU激活函数。

   该激活函数定义如下：

   .. math::
       \text{RReLU}(x_{ji}) = \begin{cases}x_{ji}, &\text{if } x_{ji} \geq 0; \cr
       {\alpha_{ji}} * x_{ji}, &\text{otherwise.}\end{cases}

   其中，:math:`\alpha_{ji}` ~ :math:`U(l, u)`, :math:`l \le u`.

   更多细节详见 `Empirical Evaluation of Rectified Activations in Convolution Network <https://arxiv.org/pdf/1505.00853.pdf>`_。

   参数：
       - **lower** (Union[int, float]) - x<0时激活函数的斜率的下界，默认值：1/8。
       - **upper** (Union[int, float]) - x<0时激活函数的斜率的上界，默认值：1/3。

   输入：
       - **x** （Tensor） - 计算RReLU的任意维度的Tensor。

   输出：
       Tensor，数据类型和shape与 `x` 相同。

   异常：
       - **TypeError** - `lower` 不是浮点数或整数。
       - **TypeError** - `upper` 不是浮点数或整数。
       - **TypeError** - `x` 不是Tensor。
       - **TypeError** - `x` 内的数据类型不是mindspore.float16或mindspore.float32。
       - **ValueError** - `lower` 大于 `upper`。
