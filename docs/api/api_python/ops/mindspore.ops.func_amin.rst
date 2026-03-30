mindspore.ops.amin
==================

.. py:function:: mindspore.ops.amin(input, axis=None, keepdims=False, *, initial=None, where=None)

    返回tensor在指定轴上的最小值。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - 参数 `axis` 将重命名为 `dim` ，默认值变更为 ``()`` 。
        - 参数 `keepdims` 将重命名为 `keepdim` 。
        - 关键字参数 `initial` 和 `where` 将被删除。

        变更后的接口签名为 ``mindspore.ops.amin(input, dim=(), keepdim=False)`` 。

    参数：
        - **input** (Tensor[Number]) - 输入tensor。
        - **axis** (Union[int, tuple(int), list(int), Tensor], 可选) - 指定计算的轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。2.9.0（不含）之后版本将重命名为 `dim` 。
        - **keepdims** (bool, 可选) - 输出tensor是否保留维度。默认 ``False`` 。2.9.0（不含）之后版本将重命名为 `keepdim` 。

    关键字参数：
        - **initial** (scalar, 可选) - 最小值的初始值。默认 ``None`` 。2.9.0（不含）之后版本将删除此参数。
        - **where** (Tensor[bool], 可选) - 指定计算最小值的范围，该tensor的shape须可被广播到 `input` 的shape上。必须指定initial值。默认 ``None`` ，表示计算全部元素。2.9.0（不含）之后版本将删除此参数。

    返回：
        Tensor
