mindspore.ops.CTCLossV2
=======================

.. py:class:: mindspore.ops.CTCLossV2(blank=0, reduction="none", zero_infinity=False)

    计算CTC(Connectionist Temporal Classification)损失和梯度。

    CTC算法是在 `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data with Recurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ 中提出的。 

    参数：
        - **blank** (int，可选) - 空白标签。默认值：0。
        - **reduction** (str，可选) - 对输出应用特定的缩减方法。目前仅支持“none”，不区分大小写。默认值：“none”。
        - **zero_infinity** (bool，可选) - 是否将无限损失和相关梯度设置为零。默认值：False。

    输入：
        - **log_probs** (Tensor) - 输入Tensor，其shape为 :math:`(T, C, N)` 的三维Tensor。 :math:`T` 表示输入长度， :math:`N` 表示批大小， :math:`C` 表示类别数，包含空白标签。
        - **targets** (Tensor) - 标签序列。其shape为 :math:`(N, S)` 的三维Tensor。 :math:`S` 表示最大标签长度。
        - **input_lengths** (Union(Tuple, Tensor)) - 输入的长度。其shape为 :math:`(N)` 。
        - **target_lengths** (Union(Tuple, Tensor)) - 标签的长度。其shape为 :math:`(N)` 。

    输出：
        - **neg_log_likelihood** (Tensor) - 相对于每个输入节点可微分的损失值。
        - **log_alpha** (Tensor) - 输入到目标的可能跟踪概率。

    异常：
        - **TypeError** - 如果 `zero_infinity` 不是bool类型。
        - **TypeError** - 如果 `reduction` 不是string类型。
        - **TypeError** - 如果 `log_probs` 的dtype不是float类型或double类型。
        - **TypeError** - 如果 `targets`、 `input_lengths` 或 `target_lengths` 的dtype不是int32类型或int64类型。
        - **ValueError** - 如果 `log_probs` 的秩不等于2。
        - **ValueError** - 如果 `targets` 的秩不等于2。
        - **ValueError** - 如果 `input_lengths` 的shape与批大小 :math:`N` 不匹配。
        - **ValueError** - 如果 `targets` 的shape与批大小 :math:`N` 不匹配。
        - **TypeError** - 如果 `targets`、 `input_lengths` 或 `target_lengths` 的类型不同。
        - **ValueError** - 如果 `blank` 的数值不是介于0和 :math:`C` 之间。
        - **RuntimeError** - `labels_indices` 的数据类型不是int64。
        - **RuntimeError** - 如果任何 `target_lengths[i]` 不在范围 [0, `input_length[i]`] 范围内。
