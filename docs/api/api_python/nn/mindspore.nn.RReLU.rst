mindspore.nn.RReLU
==================

.. py:class:: mindspore.nn.RReLU(lower=0.125, upper=float(1. / 3))

   Randomized Leaky ReLU激活函数。

   该激活函数定义如下：

   .. math::
      \text{RReLU}(x_{ji}) = \begin{cases}x_{ji}, &\text{if } x_{ji} \geq 0; \cr
        {\alpha_{ji}} * x, &\text{otherwise.}\end{cases}

   其中，:math:`\alpha_{ji}` ~ U(l, u), :math:`l \le u`.

   更多细节详见 `https://arxiv.org/pdf/1505.00853.pdf`_。

   **参数：**

   - **lower** (`Union[int, float]`) - x<0时激活函数的斜率的下界，默认值：0.125。
   - **upper** (`Union[int, float]`) - x<0时激活函数的斜率的上界，默认值：1/3。

   **输入：**

   - **x** （Tensor） - 计算RReLU的任意维度的Tensor。

   **输出：**

   Tensor，数据类型和shape与 `x` 相同。

   **异常：**

   - **TypeError** - `lower` 不是浮点数或整数。
   - **TypeError** - `upper` 不是浮点数或整数。
   - **ValueError** - `lower` 大于 `upper`。
