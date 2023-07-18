mindspore.nn.L1Loss
=============================

.. py:class:: mindspore.nn.L1Loss(reduction='mean')

    L1Loss用于计算预测值和目标值之间的平均绝对误差。
    
    假设 :math:`x` 和 :math:`y` 为一维Tensor，长度 :math:`N` ，则计算 :math:`x` 和 :math:`y` 的loss而不进行降维操作（即reduction参数设置为"none"）。

    公式如下：
    
    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|,

    其中， :math:`N` 为batch size。如果 `reduction` 不是"none"，则：

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    参数：
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    输入：
        - **logits** (Tensor) - 预测值，任意维度的Tensor。
        - **labels** (Tensor) - 目标值，通常情况下与 `logits` 的shape相同。但是如果 `logits` 和 `labels` 的shape不同，需要保证他们之间可以互相广播。

    输出：
        Tensor，类型为float。

    异常：
        - **ValueError** - `reduction` 不为"mean"、"sum"或"none"。
        - **ValueError** - `logits` 和 `labels` 有不同的shape，且不能互相广播。
