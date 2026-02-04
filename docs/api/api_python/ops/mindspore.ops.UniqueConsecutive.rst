mindspore.ops.UniqueConsecutive
================================

.. py:class:: mindspore.ops.UniqueConsecutive(return_inverse=False, return_counts=False, dim=None)

    对输入Tensor中连续且重复的元素去重。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.unique_consecutive` 。

    参数：
        - **return_inverse** (bool, 可选) - 是否返回每个输入中元素映射到输出中位置的索引。默认值： ``False`` 。
        - **return_counts** (bool, 可选) - 是否返回每个去重元素在输入中所在的连续序列的计数。默认值： ``False`` 。
        - **dim** (int, 可选) - 维度。如果为 ``None`` ，则对输入进行展平操作。如果指定，必须是int32或int64类型。默认值： ``None`` 。

    输入：
        - **x** (Tensor) - 输入Tensor。

    输出：
        Tensor或包含Tensor对象的元组（ `output` 、 `idx` 、 `counts` ）。

        - `output` 为去重后的输出，与 `x` 具有相同的数据类型。
        - 如果 `return_inverse` 为 ``True`` ，则返回Tensor `idx` ，shape与 `x` 相同，表示每个输入中元素映射到输出中位置的索引。
        - 如果 `return_counts` 为 ``True`` ，则返回Tensor `counts` ，表示每个去重元素在输入中所在的连续序列的计数。
