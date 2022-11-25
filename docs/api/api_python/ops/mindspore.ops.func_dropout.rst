mindspore.ops.dropout
======================

.. py:function:: mindspore.ops.dropout(x, p=0.5, seed0=0, seed1=0)

    在训练期间，以服从伯努利分布的概率 `p` 随机将输入Tensor的某些值归零，起到减少神经元相关性的作用，避免过拟合。

    参数：
        - **x** (Tensor) -  dropout的输入，任意维度的Tensor，其数据类型为float16或float32。
        - **p** (float) - 输入神经元丢弃概率，数值范围在0到1之间。例如，p=0.1，删除10%的神经元。默认值：0.5。
        - **seed0** (int) - 算子层的随机种子，用于生成随机数。默认值：0。
        - **seed1** (int) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。默认值：0。

    返回：
        - **output** (Tensor) - shape和数据类型与 `x` 相同。
        - **mask** (Tensor) - shape与 `x` 相同。

    异常：
        - **TypeError** - `p` 不是float。
        - **TypeError** - `seed0` 或 `seed1` 不是int。
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **TypeError** - `x` 不是Tensor。