mindspore.ops.shuffle
=====================

.. py:function:: mindspore.ops.shuffle(x, seed=None)

    沿着第一维随机打乱tensor。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **x** (Tensor) - 输入tensor。
        - **seed** (int, 可选) - 随机种子，默认 ``None`` ，等同于 ``0`` 。

    返回：
        Tensor
