mindspore.ops.Dropout
======================

.. py:class:: mindspore.ops.Dropout(keep_prob=0.5, Seed0=0, Seed1=0)

    Dropout是一种正则化手段，通过在训练中以 :math:`1 - keep\_prob` 的概率随机将神经元输出设置为0，起到减少神经元相关性的作用，避免过拟合。

    **参数：**

    - **keep_prob** (float) - 输入神经元保留概率，数值范围在0到1之间。例如，keep_prob=0.9，删除10%的神经元。默认值：0.5。
    - **Seed0** (int) - 算子层的随机种子，用于生成随机数。默认值：0。
    - **Seed1** (int) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。默认值：0。

    **输入：**

    - **x** (Tensor) -  Dropout的输入，任意维度的Tensor，其数据类型为float16或float32。

    **输出：**

    - **output** (Tensor) - shape和数据类型与 `x` 相同。
    - **mask** (Tensor) - shape与 `x` 相同。

    **异常：**

    - **TypeError** - `keep_prob` 不是float。
    - **TypeError** - `Seed0` 或 `Seed1` 不是int。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。
    - **TypeError** - `x` 不是Tensor。