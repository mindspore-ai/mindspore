mindspore.ops.KLDivLoss
=======================

.. py:class:: mindspore.ops.KLDivLoss(reduction='mean')

    计算输入 `logits` 和 `labels` 的KL散度。

    KLDivLoss的计算公式如下：

    .. math::
        L = \{l_1,\dots,l_N\}^\top, \quad
        l_n = target_n \cdot (\log target_n - x_n)

    可得

    .. math::
        \ell(x, target) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{batchmean}(L), & \text{if reduction} = \text{'batchmean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    其中 :math:`x` 代表 `logits` ；
    :math:`target` 代表 `labels` ；
    :math:`\ell(x, target)` 为 `output` 。

	.. note::
        目前Ascend平台不支持设置 `reduction` 为 "mean"。
        目前Ascend平台不支持数据类型float64。
        目前GPU平台不支持设置 `reduction` 为 "batchmean"。
        仅当 `reduction` 设置为"batchmean"时输出才符合该数学公式。


    **参数：**
    
    - **reduction** (str) - 指定输出结果的计算方式。可选值为："none"、"mean"、"batchmean"或"sum"。 默认值: "mean"。

    **输入：**
    
    - **logits** (Tensor) - 数据类型支持float16、float32或float64。
    - **labels** (Tensor) - 标签Tensor，与 `logits` 的shape和数据类型相同。

    **输出：**
    
    Tensor或标量。如果 `reduction` 为 'none' ，则输出为Tensor且与 `logits` 的shape相同。否则为标量。

    **异常：**
    
    - **TypeError** - `reduction` 不是str。
    - **TypeError** - `logits` 或 `labels` 不是Tensor。
    - **TypeError** - `logits` 或 `labels` 的数据类型不是float32。
