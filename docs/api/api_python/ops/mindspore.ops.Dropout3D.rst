mindspore.ops.Dropout3D
========================

.. py:class:: mindspore.ops.Dropout3D(keep_prob=0.5)

    随机丢弃层。

    Dropout是一种正则化手段，该算子根据丢弃概率 :math:`1 - keep\_prob` ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合，在推理过程中，此层返回与 `x` 相同的Tensor。对于shape为NCDHW的五维Tensor，通道特征图指的是shape为DHW的三维特征图。

    例如，输入的批数据中第 :math:`i_th` 个样本的第 :math:`j_th` 个通道，则可三维Tensor input[i,j,k]。

    Dropout3D可以提高通feature map之间的独立性。

    **参数：**

    keep_prob (float)：输入通道保留率，数值范围在0到1之间，例如 `keep_prob` = 0.8，意味着过滤20%的通道。默认值：0.5。

    **输入：**

    - **x** (Tensor) - shape为 :math:`(N, C, D, H, W)` 的五维Tensor。N代表批次大小，C代表管道，D代表特征深度，H代表特征高度，W代表特征宽度。数据类型为int8、int16、int32、int64、float16或float32。

    **输出：**

    - **output** (Tensor) - shape和数据类型与 `x` 相同。
    - **mask** (Tensor) - shape与 `x` 相同，数据类型为bool。

    **异常：**

    - **TypeError** - `keep_prob` 的数据类型不是float。
    - **ValueError** - `keep_prob` 超出[0.0, 1.0]范围，或者输入的维度不是5。