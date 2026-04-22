mindspore.ops.slice_scatter
===========================

.. py:function:: mindspore.ops.slice_scatter(input, src, axis=0, start=None, end=None, step=1)

    沿指定轴将源tensor嵌入到切片后的目标tensor。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **src** (Tensor) - 用于嵌入到 `input` 的源tensor。
        - **axis** (int，可选) - 指定轴，默认 ``0`` 。
        - **start** (int，可选) - 指定轴中，嵌入的开始索引，默认 ``None`` ，表示 `start` 为 ``0`` 。
        - **end** (int，可选) - 指定轴中，嵌入的结束索引，默认 ``None`` ，表示 `end` 是 `input` 在指定维度的长度。
        - **step** (int，可选) - 嵌入时跳过的步长，默认 ``1`` 。

    返回：
        Tensor
