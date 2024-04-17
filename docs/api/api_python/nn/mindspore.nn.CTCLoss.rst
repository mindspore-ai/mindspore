mindspore.nn.CTCLoss
====================

.. py:class:: mindspore.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

    CTCLoss损失函数。主要用来计算连续未分段的时间序列与目标序列之间的损失。

    关于CTCLoss算法详细介绍，请参考 `Connectionist Temporal Classification: Labeling Unsegmented Sequence Data withRecurrent Neural Networks <http://www.cs.toronto.edu/~graves/icml_2006.pdf>`_ 。

    参数：
        - **blank** (int，可选) - 空白标签。默认值：``0`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``"none"`` 、 ``"mean"`` 、 ``"sum"`` ，默认值： ``"mean"`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

        - **zero_infinity** (bool，可选) - 在损失无限大的时候，是否将无限损失和相关梯度置为零。默认值： ``False`` 。

    输入：
        - **log_probs** (Tensor) - 预测值，shape为 :math:`(T, N, C)` 或 :math:`(T, C)` 。其中T表示输入长度，N表示批次大小，C是分类数。T，N，C均为正整数。
        - **targets** (Tensor) - 目标值，shape为 :math:`(N, S)` 或 (sum( `target_lengths` ))。其中S表示最大目标长度。
        - **input_lengths** (Union[tuple, Tensor]) - shape为 :math:`(N)` 的Tensor或tuple。表示输入长度。
        - **target_lengths** (Union[tuple, Tensor]) - shape为 :math:`(N)` 的Tensor或tuple。表示目标长度。

    输出：
        - **neg_log_likelihood** (Tensor) - 对每一个输入节点可微调的损失值。

    异常：
        - **TypeError** - `log_probs` 或 `targets` 不是Tensor。
        - **TypeError** - `zero_infinity` 不是布尔值， `reduction` 不是字符串。
        - **TypeError** - `log_probs` 的数据类型不是float或double。
        - **TypeError** - `targets` ， `input_lengths` 或 `target_lengths` 数据类型不是int32或int64。
        - **ValueError** - `reduction` 不为 ``"none"`` ， ``"mean"`` 或 ``"sum"`` 。
        - **ValueError** - `blank` 值不介于0到C之间。C是 `log_probs` 的分类数。
        - **ValueError** - 当 `log_prob` 的shape是 :math:`(T, C)` 时， `target` 的维度不是1或2。
        - **ValueError** - 当 `log_prob` 的shape是 :math:`(T, C)` 时， `target` 的首个维度的长度不是1。
        - **RuntimeError** - `input_lengths` 的值大于T。T是 `log_probs` 的长度。
        - **RuntimeError** - `target_lengths[i]` 的值不介于0到 `input_length[i]` 之间。
