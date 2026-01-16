mindspore.ops.kron
==================

.. py:function:: mindspore.ops.kron(input, other)

    计算 `input` 和 `other` 的Kronecker积。

    如果 `input` 的shape为 :math:`(r0, r1, ... , rN)` ， `other` 的shape为 :math:`(s0, s1, ... , sN)` ，则计算结果的shape为 :math:`(r0 * s0, r1 * s1, ... , rN * sN)` 。

    .. math::
        (input ⊗ y)_{k_{0},k_{1},...k_{n}} =
        input_{i_{0},i_{1},...i_{n}} * other_{j_{0},j_{1},...j_{n}},

    其中，对于所有的 0 ≤ `t` ≤ `n`，都有 :math:`k_{t} = i_{t} * b_{t} + j_{t}` 。

    .. note::
        支持实数和复数类型的输入。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **other** (Tensor) - 输入tensor。

    返回：
        Tensor