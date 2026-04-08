mindspore.ops.GLU
=================

.. py:class:: mindspore.ops.GLU(axis=-1)

    计算输入Tensor的门线性单元激活函数（Gated Linear Unit activation function）值。

    .. math::
        {GLU}(a, b)= a \otimes \sigma(b)

    其中，:math:`a` 表示输入Tensor `x` 拆分后的前一半元素，:math:`b` 表示 `x` 拆分后的另一半元素。
    这里 :math:`\sigma` 为sigmoid函数，:math:`\otimes` 是Hadamard乘积。
    请参考 `Language Modeling with Gated Convolutional Networks <https://arxiv.org/abs/1612.08083>`_ 。

    参数：
        - **axis** (int，可选) - 指定分割输入 `x` 的轴，取值范围 `[-r, r)` ，其中 `r` 为 `x` 的维度数。默认值： ``-1`` ，输入 `x` 的最后一维。

    输入：
        - **x** (Tensor) - 待计算的Tensor，数据类型为浮点型，shape为 :math:`(\ast_1, N, \ast_2)` ，其中 `*` 为任意额外维度，且要求 :math:`N` 为偶数。 :math:`N` 为 `x` 在被 `axis` 选中的维度上的大小。

    输出：
        Tensor，数据类型与输入 `x` 相同，shape为 :math:`(\ast_1, M, \ast_2)`，其中 :math:`M=N/2` 。

    异常：
        - **TypeError** - `x` 不是Tensor或 `axis` 不是整型。
        - **IndexError** - `axis` 的值超出了 `[-r, r)` 的范围，其中 `r` 为 `x` 的维度数。
        - **RuntimeError** - `x` 数据类型不被支持。
        - **RuntimeError** - `x` 在被 `axis` 选中的维度上的长度不为偶数。
