mindspore.nn.LeakyReLU
=======================

.. py:class:: mindspore.nn.LeakyReLU(alpha=0.2)

    逐元素计算Leaky ReLU激活函数。

    该激活函数定义如下：

    .. math::
        \text{leaky_relu}(x) = \begin{cases}x, &\text{if } x \geq 0; \cr
        {\alpha} * x, &\text{otherwise.}\end{cases}

    其中，:math:`\alpha` 表示 `alpha` 参数。

    更多细节详见 `Rectifier Nonlinearities Improve Neural Network Acoustic Models <https://ai.stanford.edu/~amaas/papers/relu_hybrid_icml2013_final.pdf>`_ 。

    LeakyReLU函数图：

    .. image:: ../images/LeakyReLU.png
        :align: center

    参数：
        - **alpha** (`Union[int, float]`) - `x` 小于0时激活函数的斜率，默认值： ``0.2`` 。

    输入：
        - **x** （Tensor） - 计算LeakyReLU的任意维度的Tensor。

    输出：
        Tensor，数据类型和shape与 `x` 相同。

    异常：
        - **TypeError** - `alpha` 不是浮点数或整数。
