mindspore.mint.nn.functional.dropout
====================================

.. py:function:: mindspore.mint.nn.functional.dropout(input, p=0.5, training=True)

    在训练期间，以服从伯努利分布的概率 `p` 随机将输入Tensor的某些值归零，起到减少神经元相关性的作用，避免过拟合。并且训练过程中返回值会乘以 :math:`\frac{1}{1-p}` 。在推理过程中，此层返回与 `input` 相同的Tensor。

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(*, N)` 。
        - **p** (float) - 输入神经元丢弃概率，数值范围在0到1之间。例如， `p` =0.1，删除10%的神经元。默认值： ``0.5`` 。
        - **training** (bool) - 若为 ``True`` 则启用dropout功能，若为 ``False`` 则直接返回输入，并且 `p` 无效。默认值： ``True`` 。

    返回：
        - **output** (Tensor) - 归零后的Tensor，shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `p` 不是float。
        - **TypeError** - `input` 不是Tensor。
