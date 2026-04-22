mindspore.ops.dropout
======================

.. py:function:: mindspore.ops.dropout(input, p=0.5, training=True, seed=None)

    在训练期间，以服从伯努利分布的概率 `p` 随机将输入Tensor的某些值归零，起到减少神经元相关性的作用，避免过拟合。并且训练过程中返回值会乘以 :math:`\frac{1}{1-p}` 。在推理过程中，此层返回与 `input` 相同的Tensor。

    .. warning::
        - 从2.9.0（不含）之后版本开始，`seed` 参数将删除，并新增 `inplace` 参数。
        - Ascend后端不支持随机数重现功能， `seed` 参数不起作用。

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(*, N)` ，数据类型为float16、float32或float64。
        - **p** (float，可选) - 输入神经元丢弃概率，数值范围在0到1之间。例如，p=0.1，删除10%的神经元。默认 ``0.5`` 。
        - **training** (bool，可选) - 若为 ``True`` 则启用dropout功能。默认 ``True`` 。
        - **seed** (int, 可选) - 随机数生成器的种子，必须是非负数，默认 ``None`` ，即 `seed` 的数量为 ``0`` 。

    返回：
        - **output** (Tensor) - 归零后的Tensor，shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `p` 不是float。
        - **TypeError** - `input` 的数据类型不是float16、float32或float64。
        - **TypeError** - `input` 不是Tensor。
