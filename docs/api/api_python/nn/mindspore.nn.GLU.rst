mindspore.nn.GLU
=================

.. py:class:: mindspore.nn.GLU(axis=-1)

    门线性单元函数（Gated Linear Unit function）。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `axis` 将重命名为 `dim`。

    .. math::
        {GLU}(a, b)= a \otimes \sigma(b)


    其中，:math:`a` 表示输入Tensor的前一半元素，:math:`b` 表示输入Tensor的另一半元素。
    这里 :math:`\sigma` 为sigmoid函数，:math:`\otimes` 是Hadamard乘积。

    参数：
        - **axis** (int，可选) - 指定分割轴。数据类型为整型，默认值： ``-1`` ，输入x的最后一维。

    输入：
        - **x** (Tensor) - Tensor的shape为 :math:`(\ast_1, N, \ast_2)` 。 `*` 表示任意数量的维度。

    输出：
        Tensor，数据类型与输入 `x` 相同，shape为 :math:`(\ast_1, M, \ast_2)`，其中 :math:`M=N/2` 。
