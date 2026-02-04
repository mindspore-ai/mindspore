mindspore.mint.nn.GLU
=====================

.. py:class:: mindspore.mint.nn.GLU(dim=-1)

    计算输入Tensor的门线性单元激活函数（Gated Linear Unit activation function）值。

    .. math::
        {GLU}(a, b)= a \otimes \sigma(b)

    其中，:math:`a` 表示输入Tensor `input` 拆分后的前一半元素，:math:`b` 表示 `input` 拆分后的另一半元素。
    这里 :math:`\sigma` 为sigmoid函数，:math:`\otimes` 是逐元素乘积。
    请参考 `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_ 。

    参数：
        - **dim** (int，可选) - 指定分割 `input` 的维度，取值范围 `[-r, r)` ，其中 `r` 为 `input` 的维度数。默认 ``-1`` ，即输入 `input` 的最后一维。

    输入：
        - **input** (Tensor) - 待计算的Tensor，数据类型为浮点型，shape为 :math:`(\ast_1, N, \ast_2)` ，其中 `*` 为任意额外维度，且要求 :math:`N` 为偶数。 :math:`N` 为 `input` 在被 `dim` 选中的维度上的大小。

    输出：
        Tensor，数据类型与输入 `input` 相同，shape为 :math:`(\ast_1, M, \ast_2)`，其中 :math:`M=N/2` 。

    异常：
        - **IndexError** - `dim` 的值超出了 `[-r, r)` 的范围，其中 `r` 为 `input` 的维度数。
        - **RuntimeError** - `input` 数据类型不被支持。
        - **RuntimeError** - `input` 在被 `dim` 选中的维度上的长度不为偶数。
