mindspore.mint.nn.functional.leaky_relu
=======================================

.. py:function:: mindspore.mint.nn.functional.leaky_relu(input, negative_slope=0.01)

    leaky_relu激活函数。 `input` 中小于0的元素乘以 `negative_slope` 。

    该激活函数定义如下：

    .. math::
        \text{leaky_relu}(input) = \begin{cases}input, &\text{if } input \geq 0; \cr
        {\negative_slope} * input, &\text{otherwise.}\end{cases}

    其中，:math:`\negative_slope` 表示 `negative_slope` 参数。

    更多细节详见 `Rectifier Nonlinearities Improve Neural Network Acoustic Models <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_ 。

    LeakyReLU函数图：

    .. image:: ../images/LeakyReLU.png
        :align: center

    参数：
        - **input** (Tensor) - 计算leaky_relu的任意维度的Tensor。
        - **negative_slope** (Union[int, float]) - `input` 的元素小于0时激活函数的斜率，默认值： ``0.01`` 。

    返回：
        Tensor，数据类型和shape与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `negative_slope` 不是浮点数或整数。
