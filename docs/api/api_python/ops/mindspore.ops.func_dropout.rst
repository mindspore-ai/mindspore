mindspore.ops.dropout
======================

.. py:function:: mindspore.ops.dropout(input, p=0.5)

    在训练期间，以服从伯努利分布的概率 `p` 随机将输入Tensor的某些值归零，起到减少神经元相关性的作用，避免过拟合。此概率与 `ops.Dropout` 和 `nn.Dropout` 中的含义相反。

    参数：
        - **input** (Tensor) - dropout的输入，任意维度的Tensor，其数据类型为float16或float32。
        - **p** (float，可选) - 输入神经元丢弃概率，数值范围在0到1之间。例如，p=0.1，删除10%的神经元。默认值：0.5。

    返回：
        - **output** (Tensor) - 归零后的Tensor，shape和数据类型与 `input` 相同。
        - **mask** (Tensor) - 用于归零的掩码，内部会按位压缩与对齐。

    异常：
        - **TypeError** - `p` 不是float。
        - **TypeError** - `input` 的数据类型既不是float16也不是float32。
        - **TypeError** - `input` 不是Tensor。