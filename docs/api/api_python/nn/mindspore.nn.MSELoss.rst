mindspore.nn.MSELoss
=============================

.. py:class:: mindspore.nn.MSELoss(reduction='mean')

    用于计算预测值与标签值之间的均方误差。
    
    假设 :math:`x` 和 :math:`y` 为一维Tensor，长度 :math:`N` ，则计算 :math:`x` 和 :math:`y` 的unreduced loss（即reduction参数设置为"none"）的公式如下：
    
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with} \quad l_n = (x_n - y_n)^2.

    其中， :math:`N` 为batch size。如果 `reduction` 不是"none"，则：

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    参数：
        - **reduction** (str) - 应用于loss的reduction类型。取值为"mean"，"sum"，或"none"。默认值："mean"。

    输入：
        - **logits** (Tensor) - 输入预测值，任意维度的Tensor。
        - **labels** (Tensor) - 输入标签，任意维度的Tensor，在通常情况下与 `logits` 的shape相同。但是如果 `logits` 和 `labels` 的shape不同，需要保证他们之间可以互相广播。

    输出：
        Tensor，为float类型的loss，如果 `reduction` 为"mean"或"sum"，则shape为0；
        如果 `reduction` 为"none"，则输出的shape为输入Tensor广播后的shape。

    异常：
        - **ValueError** - `reduction` 不为"mean"，"sum"，或"none"。
        - **ValueError** - `logits` 和 `labels` 的shape不同，且不能广播。
