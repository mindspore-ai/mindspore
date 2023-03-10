mindspore.ops.multi_margin_loss
================================

.. py:function:: mindspore.ops.multi_margin_loss(input, target, p=1, margin=1, weight=None, reduction='mean')

    用于优化多类分类问题的合页损失。

    优化输入和输出之间的多级分类合页损耗（基于边缘损失）。

    对于每个小批量样本，1D输入 :math:`x` 和标量输出 :math:`y` 的损失为：

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, \text{margin} - x[y] + x[i])^p}{\text{x.size}(0)}

    其中 :math:`i\in \{0,⋯,x.size(0)−1\}` 并且 :math:`i \ne y`。

    参数：
        - **input** (Tensor) - 输入，shape为 :math:`(N, C)`。数据类型只支持float32、float16或float64。即上述公式中的 :math:`x` 。
        - **target** (Tensor) - 真实标签，shape为 :math:`(N,)`。数据类型只支持int64。值应为非负值，且小于C。即上述公式中的 :math:`y` 。
        - **p** (int, 可选) - 对偶距离的范数度。必须为1或2。默认值：1。
        - **margin** (int, 可选) - 改变对偶距离的参数。默认值：1。
        - **weight** (Tensor, 可选) - 每个类别的缩放权重，shape为 :math:`(C,)`。数据类型只支持float32、float16或float64。默认值：None。
        - **reduction** (str, 可选) - 对输出应用特定的缩减方法：可选"none"、"mean"、"sum"。默认值：'mean'。

          - 'none'：不应用缩减方法。
          - 'mean'：输出的值总和除以输出的元素个数。
          - 'sum'：输出的总和。

    返回：
        - **outputs** - 当 `reduction` 为"none"时，类型为Tensor，shape和 `target` 相同。否则，为标量。

    异常：
        - **TypeError** - `p` 或者 `target` 数据类型不是int。
        - **TypeError** - `margin` 数据类型不是int。
        - **TypeError** - `reduction` 数据类型不是str。
        - **TypeError** - `input` 数据类型不是以下之一：float16、float、float64。
        - **TypeError** - `weight` 和 `input` 的数据类型不相同。
        - **ValueError** - `p` 的值不是以下之一：1、2。
        - **ValueError** - `reduction` 的值不是以下之一：{"none","sum","mean"}。
        - **ValueError** - `input` 的shape[0]和 `target` 的shape[0]不相等。
        - **ValueError** - `input` 的shape[1]和 `weight` 的shape[0]不相等。
        - **ValueError** - 如果有以下情形： `weight` 的维度不是1、 `target` 的维度不是1、 `input` 的维度不是2。
