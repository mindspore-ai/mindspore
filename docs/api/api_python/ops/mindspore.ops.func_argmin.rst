mindspore.ops.argmin
====================

.. py:function:: mindspore.ops.argmin(input, axis=None, keepdims=False)

    返回tensor在指定轴上的最小值索引。

    .. warning::
        从2.9.0（不含）之后版本开始，参数 `axis` 将重命名为 `dim`，`keepdims` 将重命名为 `keepdim`，
        接口签名将变更为 ``mindspore.ops.argmin(input, dim=None, keepdim=False)`` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (Union[int, None]，可选) - 指定轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **keepdims** (bool，可选) - 输出tensor是否保留维度。默认 ``False`` 。

    返回：
        Tensor
