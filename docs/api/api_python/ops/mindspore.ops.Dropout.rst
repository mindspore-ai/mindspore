mindspore.ops.Dropout
======================

.. py:class:: mindspore.ops.Dropout(keep_prob=0.5, Seed0=0, Seed1=0)

    Dropout是一种正则化手段，通过在训练中以 :math:`1 - keep\_prob` 的概率随机将神经元输出设置为0，起到减少神经元相关性的作用，避免过拟合。

    更多细节请参考 :func:`mindspore.ops.dropout` 。

    参数：
        - **keep_prob** (float，可选) - 输入神经元保留概率，数值范围在0到1之间。例如，keep_prob=0.9，删除10%的神经元。默认值： ``0.5`` 。
        - **Seed0** (int，可选) - 算子层的随机种子，用于生成随机数。默认值： ``0`` 。
        - **Seed1** (int，可选) - 全局的随机种子，和算子层的随机种子共同决定最终生成的随机数。默认值： ``0`` 。

    输入：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(*, N)`，数据类型为float16、float32或float64。

    输出：
        - **output** (Tensor) - shape和数据类型与 `x` 相同。
        - **mask** (Tensor) - 应用于 `x` 的掩码。
        
          - 在GPU和CPU上， `mask` 具有与 `x` 相同的shape和数据类型。
          - 在Ascend上，为了获得更好的性能，它被表示为一个具有Uint8数据类型的一维Tensor。其shape为 :math:`(byte\_counts, )` ， 其中 :math:`byte\_counts` 为覆盖 `x` 的shape所需的字节数。通过下面的公式计算其大小：

            .. math::

                byte\_counts = \text{ceil}(\text{cumprod}(x.shape) / 128) * 16

            若 `x` 的shape为 :math:`(2, 3, 4, 5, 6)` ，则 `mask` 的shape为 :math:`(96, )` 。
