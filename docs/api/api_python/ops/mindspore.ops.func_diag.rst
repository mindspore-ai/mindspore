mindspore.ops.diag
===================

.. py:function:: mindspore.ops.diag(input)

    返回一个用输入tensor作为对角线，其余元素为 ``0`` 的tensor。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选参数 `diagonal` ，接口签名将变更为
        ``mindspore.ops.diag(input, diagonal=0)`` 。
        该变更与当前版本兼容，不使用 `diagonal` 参数时行为不变。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        Tensor
