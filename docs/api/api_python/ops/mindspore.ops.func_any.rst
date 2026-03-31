mindspore.ops.any
=================

.. py:function:: mindspore.ops.any(input, axis=None, keep_dims=False)

    检查指定轴上是否含有 `True`。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - 参数 `axis` 将重命名为 `dim` 。
        - 参数 `keep_dims` 将重命名为 `keepdim` 。

        变更后的接口签名为 ``mindspore.ops.any(input, dim=None, keepdim=False)`` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor], 可选) - 指定轴。如果为 ``None``，检查所有元素。默认 ``None`` 。2.9.0（不含）之后版本将重命名为 `dim` 。
        - **keep_dims** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。2.9.0（不含）之后版本将重命名为 `keepdim` 。

    返回：
        Tensor
