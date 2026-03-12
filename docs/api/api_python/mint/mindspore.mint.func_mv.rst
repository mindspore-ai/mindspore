mindspore.mint.mv
==================

.. py:function:: mindspore.mint.mv(input, vec)

    实现矩阵 `input` 和向量 `vec` 相乘。
    如果 `input` 是shape为 :math:`(N,M)` 的tensor， `vec` 是shape为 :math:`(M,)` 的tensor，
    则输出shape为 :math:`(N,)` 的一维tensor。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入矩阵。其shape为 :math:`(N,M)` ，且rank必须为二维。
        - **vec** (Tensor) - 输入向量。其shape为 :math:`(M,)` ，且rank必须为一维。

    返回：
        Tensor，shape为 :math:`(N,)` 。

    异常：
        - **TypeError** - 如果 `input` 或者 `vec` 不是Tensor。
        - **TypeError** - 如果 `input` 或 `vec` 的dtype均不是float16、float32。
        - **TypeError** - 如果 `input` 和 `vec` 的dtype不同。
        - **ValueError** - 如果 `input` 不是二维张量，或者 `vec` 不是一维张量。
