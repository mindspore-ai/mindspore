mindspore.ops.nanmean
=====================

.. py:function:: mindspore.ops.nanmean(input, axis=None, keepdims=False, *, dtype=None)

    忽略NaN值，按指定轴计算输入tensor的平均值。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：参数 `axis` 将重命名为 `dim` ，
        参数 `keepdims` 将重命名为 `keepdim` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int, 可选) - 指定轴，默认 ``None`` 。
        - **keepdims** (bool, 可选) - 输出tensor是否保留维度，默认 ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 指定数据类型，默认 ``None`` 。

    返回：
        Tensor
