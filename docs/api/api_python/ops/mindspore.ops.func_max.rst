mindspore.ops.max
=================

.. py:function:: mindspore.ops.max(input, axis=None, keepdims=False, *, initial=None, where=None)

    返回tensor在指定轴上的最大值及其索引。

    .. warning::
        2.9.0（不含）之后版本中，当 `axis` 为 ``None`` 时，返回值将从 tuple 变为单个 Tensor，且不再支持 `initial` / `where`。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **axis** (int) - 指定轴。如果为 ``None`` ，计算 `input` 中的所有元素。默认 ``None`` 。
        - **keepdims** (bool) - 输出tensor是否保留维度。默认 ``False`` 。

    关键字参数：
        - **initial** (scalar, 可选) - 最大值的初始值。默认 ``None`` 。
        - **where** (Tensor[bool], 可选) - 指定计算最大值的范围，该tensor的shape须可被广播到 `input` 的shape上。必须指定initial值。默认 ``None`` ，表示计算全部元素。

    返回：
        两个tensor组成的tuple(max, max_indices)。
