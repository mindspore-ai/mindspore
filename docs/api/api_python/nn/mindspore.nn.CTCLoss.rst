mindspore.nn.CTCLoss
====================

.. py:class:: mindspore.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

    CTCLoss损失函数。

    关于CTCLoss算法详细介绍，请参考 `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data withRecurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ 。

    参数：
        - **blank** (int) - 空白标签。默认值：0。
        - **reduction** (str) - 指定输出结果的计算方式。可选值为"none"、"mean"或"sum"。默认值："mean"。
        - **zero_infinity** (bool) - 是否设置无限损失和相关梯度为零。默认值：False。

    输入：
        - **log_probs** (Tensor) - 输入Tensor，shape :math:`(T, N, C)` 或 :math:`(T, C)` 。其中T表示输入长度，N表示批次大小，C是分类数。T，N，C均为正整数。
        - **targets** (Tensor) - 目标Tensor，shape :math:`(N, S)` 或 (sum( `target_lengths` ))。其中S表示最大目标长度。
        - **input_lengths** (Union[tuple, Tensor]) - shape为N的Tensor或tuple。表示输入长度。
        - **target_lengths** (Union[tuple, Tensor]) - shape为N的Tensor或tuple。表示目标长度。

    输出：
        - **neg_log_likelihood** (Tensor) - 对每一个输入节点可微调的损失值。

    异常：
        - **TypeError** - `log_probs` 或 `targets` 不是Tensor。
        - **TypeError** - `zero_infinity` 不是布尔值， `reduction` 不是字符串。
        - **TypeError** - `log_probs` 的数据类型不是float或double。
        - **TypeError** - `targets` ， `input_lengths` 或 `target_lengths` 数据类型不是int32或int64。
        - **ValueError** - `reduction` 不为"none"，"mean"或"sum"。
        - **ValueError** - `targets` ， `input_lengths` 或 `target_lengths` 的数据类型是不同的。
        - **ValueError** - `blank` 值不介于0到C之间。C是 `log_probs` 的分类数。
        - **ValueError** - 当 `log_prob` 的shape是 :math:`(T, C)` 时， `target` 的维度不等于1。
        - **RuntimeError** - `input_lengths` 的值大于T。T是 `log_probs` 的长度。
        - **RuntimeError** - `target_lengths[i]` 的值不介于0到 `input_length[i]` 之间。
