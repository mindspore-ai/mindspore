mindspore.ops.rot90
=======================

.. py:function:: mindspore.ops.rot90(input, k, dims)

    沿指定维度的平面内将n-D tensor旋转90度。
    如果k>0，旋转方向是从第一轴朝向第二轴，如果k<0，旋转方向从第二轴朝向第一轴。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `k` 和 `dims` 将补充默认值，接口签名将变更为
        ``mindspore.ops.rot90(input, k=1, dims=(0, 1))`` 。
        该变更与当前版本兼容，显式传入 `k` 和 `dims` 时行为不变。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **k** (int) - 旋转的次数。2.9.0（不含）之后版本默认值将变更为 ``1`` 。
        - **dims** (Union[list(int), tuple(int)]) - 指定维度。2.9.0（不含）之后版本默认值将变更为 ``(0, 1)`` 。

    返回：
        Tensor
