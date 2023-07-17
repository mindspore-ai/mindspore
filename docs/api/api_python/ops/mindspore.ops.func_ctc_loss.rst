mindspore.ops.ctc_loss
======================

.. py:function:: mindspore.ops.ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0, reduction="mean", zero_infinity=False)

    计算CTC（Connectist Temporal Classification）损失和梯度。

    CTC是序列标注问题中的一种损失函数，主要用于处理序列标注问题中的输入与输出标签的对齐问题。
    传统序列标注算法需要每一时刻输入与输出符号完全对齐，而CTC拓展了标签集合，添加了空元素。
    在使用拓展标签集合对序列进行标注后，所有可以通过映射函数转换为真实序列的预测序列，都是正确的预测结果，也就是说无需数据对齐处理，即可得到预测序列。
    其目标函数就是最大化所有正确的预测序列的概率和。

    关于CTCLoss算法详细介绍，请参考 `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data withRecurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ 。

    参数：
        - **log_probs** (Tensor) - 输入Tensor，shape :math:`(T, N, C)` 。其中T表示输入长度，N表示批次大小，C是分类数，包含空白。
        - **targets** (Tensor) - 目标Tensor，shape :math:`(N, S)` 。其中S表示最大目标长度。
        - **input_lengths** (Union(tuple, Tensor)) - 输入长度，shape为N的Tensor或tuple。
        - **target_lengths** (Union(tuple, Tensor)) - 目标长度，shape为N的Tensor或tuple。
        - **blank** (int，可选) - 空白标签。默认值： ``0`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

        - **zero_infinity** (bool，可选) - 是否设置无限损失和相关梯度为零。默认值： ``False`` 。

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
        - **ValueError** - `blank` 值不介于0到C之间。C是 `log_probs` 的分类数。
        - **RuntimeError** - `input_lengths` 的值大于T。T是 `log_probs` 的长度。
        - **RuntimeError** - `target_lengths[i]` 的取值范围不在0到 `input_length[i]` 之间。
