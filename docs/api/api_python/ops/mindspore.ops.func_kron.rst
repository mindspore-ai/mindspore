mindspore.ops.kron
==================

.. py:function:: mindspore.ops.kron(x, y)

    计算 `x` 和 `y` 的Kronecker积：:math:`x⊗y` 。
    如果 `x` 是Tensor :math:`(a_{0}` x :math:`a_{1}` x ... x :math:`a_{n})` ， `y` 是Tensor :math:`(b_{0}` x :math:`b_{1}` x ... x :math:`b_{n})` ，计算结果为Tensor :math:`(a_{0}*b_{0}` x :math:`a_{1}*b_{1}` x ... x :math:`a_{n}*b_{n})` ，计算公式如下：

    .. math::
        (x ⊗ y)_{k_{0},k_{1},...k_{n}} = x_{i_{0},i_{1},...i_{n}} * y_{j_{0},j_{1},...j_{n}},

    其中，对于所有的 0 ≤ `t` ≤ `n`，都有 :math:`k_{t} = i_{t} * b_{t} + j_{t}` 。如果其中一个Tensor维度小于另外一个，则在第一维补维度直到两Tensor维度相同为止。

    .. note::
        支持实数和复数类型的输入。

    参数：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(r0, r1, ... , rN)` 。
        - **y** (Tensor) - 输入Tensor，shape为 :math:`(s0, s1, ... , sN)` 。

    返回：
        Tensor，shape为 :math:`(r0 * s0, r1 * s1, ... , rN * sN)` 。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `y` 不是Tensor。
