mindspore.ops.multi_margin_loss
================================

.. py:function:: mindspore.ops.multi_margin_loss(inputs, target, p=1, margin=1, weight=None, reduction='mean')

    用于优化多类分类问题的铰链损失。

    创建一个标准，用于优化输入 :math:`x` （一个2D小批量Tensor）
    和输出 :math:`y` （一个目标类索引的1DTensor :math:`0 \leq y \leq \text{x.size}(1)-1`）
    之间的多类分类铰链损失（基于边距的损失）：
    对于每个小批量样本，1D输入的损失 :math:`x` 和标量输出 :math:`y` 是：

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] + x[i]))^p}{\text{x.size}(0)}

    其中 :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}`
    并且 :math:`i \neq y`。
    或者，通过向构造函数传递一个1D输入 `weight` 的Tensor来对类赋予不同的权重。

    参数：
        - **inputs** (Tensor) - 输入，shape为 :math:`(N, C)`。数据类型只支持float32、float16或float64。
        - **target** (Tensor) - 真实标签，shape为 :math:`(N,)`。数据类型只支持int64。值应为非负值，且小于C。
        - **p** (int, 可选) - 对偶距离的范数度。必须为1或2。默认值：1。
        - **margin** (int, 可选) - 改变对偶距离的参数。默认值：1。
        - **weight** (Tensor, 可选) - 每个类别的缩放权重，shape为 :math:`(C,)`。数据类型只支持float32、float16或float64。默认值：None。
        - **reduction** (str, 可选) - 对输出应用特定的缩减方法：可选"none"、"mean"、"sum"。默认值：'mean'。

          - 'none'：不应用缩减方法。
          - 'mean'：输出的值总和除以输出的元素个数。
          - 'sum'：输出的总和。

    返回：
        - **outputs** - (Tensor)，当 `reduction` 为"none"时，shape为 :math:`(N,)`。否则，为标量。类型与 `inputs` 相同。

    异常：
        - **TypeError** - `p` 或者 `target` 数据类型不是int。
        - **TypeError** - `margin` 数据类型不是int。
        - **TypeError** - `reduction` 数据类型不是str。
        - **TypeError** - `inputs` 数据类型不是以下之一：float16、float、float64。
        - **TypeError** - `weight` 和 `inputs` 的数据类型不相同。
        - **ValueError** - `p` 的值不是以下之一：1、2。
        - **ValueError** - `reduction` 的值不是以下之一：{"none","sum","mean"}。
        - **ValueError** - `inputs` 的shape[0]和 `target` 的shape[0]不相等。
        - **ValueError** - `inputs` 的shape[1]和 `weight` 的shape[0]不相等。
        - **ValueError** - 如果有以下情形： `weight` 的维度不是1、 `target` 的维度不是1、 `inputs` 的维度不是2。
