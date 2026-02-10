mindspore.mint.mm
=================

.. py:function:: mindspore.mint.mm(input, mat2)

    计算两个矩阵的乘积。

    如果 `input` 是一个 :math:`(n \times m)` 的tensor， `mat2` 是一个 :math:`(m \times p)` 的tensor， `out` 则会是一个 :math:`(n \times p)` 的tensor。

    .. note::
        此函数不能支持广播。若需要可广播的方法，请参考 :func:`mindspore.mint.matmul`。
        `input` 的最后一维度必须和 `mat2` 的第一维度相等。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 第一个矩阵tensor。
        - **mat2** (Tensor) - 第二个矩阵tensor。

    返回：
        Tensor
