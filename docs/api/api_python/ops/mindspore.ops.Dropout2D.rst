mindspore.ops.Dropout2D
=======================

.. py:class:: mindspore.ops.Dropout2D(keep_prob=0.5)

    在训练期间，根据概率 :math:`1 - keep\_prob` ，随机的将一些通道设置为0，且服从伯努利分布。（对于shape为 :math:`(N, C, H, W)` 的四维Tensor，通道特征图指的是shape为 :math:`(H, W)` 的二维特征图。）

    例如，对于批量输入的第 :math:`i_th` 样本的第:math:`j_th` 通道为二维Tensor，即input[i,j]。在前向传播过程中，输入样本的每个通道都有可能被置为0，置为0的概率为 :math:`1 - keep\_prob`，且服从伯努利分布。
    
    论文 `Dropout: A Simple Way to Prevent Neural Networks from Overfitting <http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf>`_ 中提出了该技术，并证明其能有效地减少过度拟合，防止神经元共适应。更多详细信息，请参见 `Improving neural networks by preventing co-adaptation of feature detectors <https://arxiv.org/pdf/1207.0580.pdf>`_ 。

    Dropout2D 可以提高通道特征图之间的独立性。

    **参数：**

    keep_prob (float) - 输入通道保留率，数值范围在0到1之间，例如 `keep_prob` = 0.8，意味着过滤20%的通道。默认值：0.5。

    **输入：**

    - **x** (Tensor) - shape为 :math:`(N, C, H, W)` 的四维张量，其中N是批量大小，C是通道数，H是特征高度，W是特征宽度。数据类型应为int8、int16、int32、int64、float16或float32。

    **输出：**

    - **output** (Tensor) - shape和数据类型与 `x` 相同。
    - **mask** (Tensor) - shape与 `x` 相同，数据类型为bool。

    **异常：**

    - **TypeError** - `keep_prob` 的数据类型不是float。
    - **ValueError** - `keep_prob` 超出[0.0, 1.0]范围，或者输入的维度不是四维。
    