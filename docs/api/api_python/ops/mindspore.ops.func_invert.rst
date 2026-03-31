mindspore.ops.invert
====================

.. py:function:: mindspore.ops.invert(x)

    逐元素按位翻转。如：01010101 变为 10101010。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. math::
        out_i = \sim x_{i}

    参数：
        - **x** (Tensor) - 输入tensor。

    返回：
        Tensor
