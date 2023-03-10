mindspore.ops.polygamma
=======================

.. py:function:: mindspore.ops.polygamma(n, input)

    计算关于 `x` 的多伽马函数的 :math:`a` 阶导数。

    .. math::
        \psi^{(a)}(x) = \frac{d^{(a)}}{dx^{(a)}} \psi(x)
    
    其中 :math:`\psi(x)` 为digamma函数。

    参数：
        - **n** (Tensor) - 多伽马函数求导的阶数，支持的数据类型为int32和int64， `n` 的shape为 :math:`()` 。
        - **input** (Tensor) - 用于计算多伽马函数的Tensor。

    返回：
        Tensor。数据类型与 `input` 一致。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不是float16、float32或float64。
        - **TypeError** - `n` 的数据类型不是int32或int64。
        - **TypeError** - `n` 的shape不是 :math:`()` 。
