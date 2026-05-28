mindspore.mint.nn.functional.mish
==================================

.. py:function:: mindspore.mint.nn.functional.mish(input)

    逐元素计算输入Tensor的MISH（A Self Regularized Non-Monotonic Neural Activation Function，自正则化非单调神经激活函数）。

    公式如下：

    .. math::
        \text{mish}(input) = input * \tanh(softplus(\text{input}))

    更多详细信息请参见 `A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_ 。

    Mish激活函数图：

    .. image:: ../images/Mish.png
        :align: center

    参数：
        - **input** (Tensor) - Mish的输入。支持数据类型：

          - Ascend：float16、float32。

    返回：
        Tensor，与 `input` 的shape和数据类型相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不是float16或float32。
