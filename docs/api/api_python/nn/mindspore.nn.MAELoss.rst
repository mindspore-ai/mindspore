mindspore.nn.MAELoss
=====================

.. py:class:: mindspore.nn.MAELoss(reduction='mean')

    衡量 :math:`x` 和 :math:`y` 之间的平均绝对误差。其中 :math:`x` 是输入 `logits` ，:math:`y` 是标签 `labels` 。
    
    简单来说，假设 :math:`x` 和 :math:`y` 是两个长度为 :math:`N` 的1D Tensor。未归约前的（参数 `reduction` 是 ``'none'``）损失为：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|

    `N` 是批次（batch）个数。

    如果 `reduction` 不是 ``'none'``，则：

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    参数：
        - **reduction** (str, 可选) - 对输出使用归约算法： ``'none'`` 、 ``'mean'`` 、 ``'sum'`` 。 默认值：``'mean'`` 。

          - ``'none'``: 不使用规约算法。
          - ``'mean'``: 计算输出的平均值。
          - ``'sum'``: 计算输出中所有元素的和。

    输入：
        - **logits** (Tensor) - Tensor的shape是 :math:`(M, *)`，其中， :math:`*` 的含义是任意额外的维度。
        - **labels** (Tensor) - Tensor的shape是 :math:`(N, *)`，通常和 `logits` 的shape相同。然而，当 `logits` 和 `labels` 的shape不同时，它们需要支持广播。

    输出：
        Tensor，加权损失，dtype是float，如果 `reduction` 是 ``'mean'`` 或 ``'sum'``，shape则为0，否则当 `reduction` 是 ``'none'`` 时，shape是广播之后的shape。

    异常：
        - **ValueError** - 如果 `reduction` 不是 ``'none'``， ``'mean'``， ``'sum'`` 中的一个。
