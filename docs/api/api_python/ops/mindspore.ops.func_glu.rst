mindspore.ops.glu
=================

.. py:function:: mindspore.ops.glu(x, axis=-1)

    门线性单元函数（Gated Linear Unit function）。

    .. math::
        {GLU}(a, b)= a \otimes \sigma(b)

    其中，:math:`a` 表示输入input_x 拆分后 Tensor的前一半元素，:math:`b` 表示输入拆分Tensor的另一半元素。:math:`\sigma` 是sigmoid函数， :math:`*` 是Hadamard乘积。
    请参考 `Language Modeling with Gated Convluational Networks <https://arxiv.org/abs/1612.08083>`_.

    参数：
        - **x** (Tensor) - 被分Tensor，数据类型为number.Number, shape为 :math:`(\ast_1, N, \ast_2)` ，其中 `*` 为任意额外维度。
        - **axis** (int，可选) - 指定分割轴。数据类型为整型，默认值：-1，输入x的最后一维。

    返回：
        Tensor，数据类型与输入 `x` 相同，shape为 :math:`(\ast_1, M, \ast_2)`，其中 :math:`M=N/2`

    异常：
        - **TypeError** -  `x` 数据类型不是number.Number。
        - **TypeError** -  `x` 不是Tensor。
