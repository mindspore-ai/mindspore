mindspore.nn.MultiMarginLoss
================================

.. py:class:: mindspore.nn.MultiMarginLoss(p=1, margin=1.0, reduction='mean', weight=None)

    多分类场景下用于计算 :math:`x` 和 :math:`y` 之间的合页损失（Hinge Loss），其中 `x` 为一个2-D Tensor，`y` 为一个表示类别索引的1-D Tensor， :math:`0 \leq y \leq \text{x.size}(1)-1`。

    对于每个小批量样本，1D输入 :math:`x` 和标量 :math:`y` 的损失为：

    .. math::
        \text{loss}(x, y) = \frac{\sum_i \max(0, w[y] * (\text{margin} - x[y] + x[i]))^p}{\text{x.size}(0)}

    其中 :math:`x \in \left\{0, \; \cdots , \; \text{x.size}(0) - 1\right\}` 并且 :math:`i \neq y`。

    参数：
        - **p** (int, 可选) - 对偶距离的范数度。必须为1或2。默认值： ``1``。
        - **margin** (float, 可选) - 改变对偶距离的参数。默认值： ``1.0`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的加权平均值。
          - ``"sum"``：计算输出元素的总和。

        - **weight** (Tensor, 可选) - 每个类别的缩放权重，shape为 :math:`(C,)`。数据类型只支持float32、float16或float64。默认值： ``None`` ，表示各个类别权重相同。

    输入：
        - **x** (Tensor) - shape为 :math:`(N, C)`。数据类型只支持float32、float16或float64。即上述公式中的 :math:`x` 。
        - **target** (Tensor) - 真实标签，shape为 :math:`(N,)`。数据类型只支持int64。值应为非负值，且小于C。 `target` 即上述公式中的 :math:`y` 。

    输出：
        Tensor，当 `reduction` 为 ``'none'`` 时，类型为Tensor，shape为 :math:`(N,)`，和 `target` 相同。否则为标量Tensor。

    异常：
        - **TypeError** - `p` 或者 `target` 数据类型不是int。
        - **TypeError** - `margin` 数据类型不是int。
        - **TypeError** - `reduction` 数据类型不是str。
        - **TypeError** - `x` 数据类型不是以下之一：float16、float、float64。
        - **TypeError** - `weight` 和 `x` 的数据类型不相同。
        - **ValueError** - `p` 的值不是以下之一：1、2。
        - **ValueError** - `reduction` 的值不是以下之一：{ ``'none'`` , ``'sum'`` , ``'mean'`` }。
        - **ValueError** - `x` 的shape[0]和 `target` 的shape[0]不相等。
        - **ValueError** - `x` 的shape[1]和 `weight` 的shape[0]不相等。
        - **ValueError** - 如果 `weight` 的维度不是1。
        - **ValueError** - 如果 `x` 的维度不是2或 `target` 的维度不是1。
