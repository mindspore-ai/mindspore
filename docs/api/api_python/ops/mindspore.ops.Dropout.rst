mindspore.ops.Dropout
======================

.. py:class:: mindspore.ops.Dropout(keep_prob=0.5, Seed0=0, Seed1=0)

     Dropout是一种正则化手段，该算子根据丢弃概率 :math:`1 - keep\_prob` ，在训练过程中随机将一些神经元输出设置为0，通过阻止神经元节点间的相关性来减少过拟合。

    **参数：**

    - **keep_prob** (float) - 输入神经元保留率，数值范围在0到1之间。例如，rate=0.9，删除10%的输入单位。默认值：0.5。
    - **Seed0** (int) - 算子层的随机种子，用于生成随机数。默认值：0。
    - **Seed1** (int) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。默认值：0。

    **输入：**

    - **x** (Tensor) -  Dropout的输入，任意维度的Tensor，其数据类型为float16或float32。

    **输出：**

    - **output** (Tensor) - shape和数据类型与 `x` 相同。
    - **mask** (Tensor) - shape与 `x` 相同。

    **异常：**

    - **TypeError** - `keep_prob` 不是float。
    - **TypeError** - `Seed0` 或`Seed1` 不是int。
    - **TypeError** - `x` 的数据类型既不是float16也不是float32。
    - **TypeError** - `x` 不是Tensor。