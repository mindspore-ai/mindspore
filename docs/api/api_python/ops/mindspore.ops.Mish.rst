mindspore.ops.Mish
====================

.. py:class:: mindspore.ops.Mish

    逐元素计算输入Tensor的MISH（Self Regularized Non-Monotonic Neural Activation Function 自正则化非单调神经激活函数）。

    公式如下：

    .. math::
        \text{output} = x * \tanh(\log(1 + \exp(\text{x})))

    更多详细信息请参见 `A Self Regularized Non-Monotonic Neural Activation Function <https://arxiv.org/abs/1908.08681>`_ 。

    输入：
        - **x** (Tensor) - shape: :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度，数据类型支持float16或float32。

    输出：
        Tensor，与 `x` 的shape和数据类型相同。

    异常：
        - **TypeError** - `x` 的数据类型非float16或float32。
