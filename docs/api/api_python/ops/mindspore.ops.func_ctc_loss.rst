mindspore.ops.ctc_loss
======================

.. py:function:: mindspore.ops.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False)

    计算CTC（Connectist Temporal Classification）损失和梯度。

    关于CTCLoss算法详细介绍，请参考 `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data withRecurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ 。

    参数：
        - **log_probs** (Tensor) - 输入Tensor，shape :math:`(T, N, C)` 。其中T表示输入长度，N表示批次大小，C是分类数，包含空白。
        - **targets** (Tensor) - 目标Tensor，shape :math:`(N, S)` 。其中S表示最大目标长度。
        - **input_lengths** (Union[tuple, Tensor]) - shape为N的Tensor或tuple。表示输入长度。
        - **target_lengths** (Union[tuple, Tensor]) - shape为N的Tensor或tuple。表示目标长度。
        - **blank** (int) - 空白标签。默认值：0。
        - **reduction** (str) - 对输出应用归约方法。可选值为"none"、"mean"或"sum"。默认值："mean"。
        - **zero_infinity** (bool) - 是否设置无限损失和相关梯度为零。默认值：False。

    返回：
        - **neg_log_likelihood** (Tensor) - 对每一个输入节点可微调的损失值，shape是 :math:`(N)`。
        - **log_alpha** (Tensor) - shape为 :math:`(N, T, 2 * S + 1)` 的输入到输出的轨迹概率。

    异常：
        - **TypeError** - `zero_infinity` 不是布尔值， `reduction` 不是字符串。
        - **TypeError** - `log_probs` 的数据类型不是float或double。
        - **TypeError** - `targets` 、 `input_lengths` 或 `target_lengths` 数据类型不是int32或int64。
        - **ValueError** - `log_probs` 的秩不是3。
        - **ValueError** - `targets` 的秩不是2。
        - **ValueError** - `input_lengths` 的shape大小不等于N。N是 `log_probs` 的批次大小。
        - **ValueError** - `target_lengths` 的shape大小不等于N。N是 `log_probs` 的批次大小。
        - **ValueError** - `targets` 、 `input_lengths` 或 `target_lengths` 的数据类型是不同的。
        - **ValueError** - `blank` 值不介于0到C之间。C是 `log_probs` 的分类数。
        - **RuntimeError** - `input_lengths` 的值大于T。T是 `log_probs` 的长度。
        - **RuntimeError** - `target_lengths[i]` 的取值范围不在0到 `input_length[i]` 之间。
