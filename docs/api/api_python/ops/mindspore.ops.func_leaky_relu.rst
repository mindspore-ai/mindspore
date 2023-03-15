mindspore.ops.leaky_relu
========================

.. py:function:: mindspore.ops.leaky_relu(input, alpha=0.2)

   leaky_relu激活函数。 `input` 中小于0的元素乘以 `alpha` 。

   该激活函数定义如下：

   .. math::
       \text{leaky_relu}(input) = \begin{cases}input, &\text{if } input \geq 0; \cr
       {\alpha} * input, &\text{otherwise.}\end{cases}

   其中，:math:`\alpha` 表示 `alpha` 参数。

   更多细节详见 `Rectifier Nonlinearities Improve Neural Network Acoustic Models <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_ 。

   参数：
       - **input** (Tensor) - 计算leaky_relu的任意维度的Tensor。
       - **alpha** (Union[int, float]) - `input` 的元素小于0时激活函数的斜率，默认值：0.2。

   返回：
       Tensor，数据类型和shape与 `input` 相同。

   异常：
       - **TypeError** - `input` 不是Tensor。
       - **TypeError** - `alpha` 不是浮点数或整数。
