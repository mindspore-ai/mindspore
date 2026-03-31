mindspore.ops.mish
==================

.. py:function:: mindspore.ops.mish(x)

    逐元素计算输入Tensor的MISH（A Self Regularized Non-Monotonic Neural Activation Function 自正则化非单调神经激活函数）。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `x` 将重命名为 `input`，并新增 `inplace` 参数。

    公式如下：

    .. math::
        \text{output} = x * \tanh(\log(1 + \exp(\text{x})))

    更多详细信息请参见 `A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_ 。

    Mish激活函数图：

    .. image:: ../images/Mish.png
        :align: center

    参数：
        - **x** (Tensor) - Mish的输入。
          支持数据类型：

          - GPU/CPU：float16、float32、float64。
          - Ascend：float16、float32。

    返回：
        Tensor，与 `x` 的shape和数据类型相同。

    异常：
        - **TypeError** - `x` 的数据类型不是float16、float32或float64。
