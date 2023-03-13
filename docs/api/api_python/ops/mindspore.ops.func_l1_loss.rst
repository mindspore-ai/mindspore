mindspore.ops.l1_loss
=====================

.. py:function:: mindspore.ops.l1_loss(input, target, reduction='mean')

    l1_loss用于计算预测值和目标值之间的平均绝对误差。

    假设 :math:`x` 和 :math:`y` 为一维Tensor，长度 :math:`N` ，`reduction` 设置为"none",则计算 :math:`x` 和 :math:`y` 的loss不进行降维操作。

    公式如下：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|,

    其中， :math:`N` 为batch size。

    如果 `reduction` 是"mean"或者"sum"，则：

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    参数：
        - **input** (Tensor) - 预测值，任意维度的Tensor。
        - **target** (Tensor) - 目标值，通常情况与 `input` 的shape相同。如果 `input` 和 `target` 的shape不同，需要保证他们之间可以互相广播。
        - **reduction** (str, optional) - 应用于loss的reduction类型。取值为"mean"，"sum"或"none"。默认值："mean"。

    返回：
        Tensor，l1_loss的结果。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `target` 不是Tensor。
        - **ValueError** - `reduction` 不为"mean"、"sum"或"none"。
