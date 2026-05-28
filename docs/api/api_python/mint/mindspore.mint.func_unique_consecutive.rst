mindspore.mint.unique_consecutive
=================================

.. py:function:: mindspore.mint.unique_consecutive(input, return_inverse=False, return_counts=False, dim=None)

    对输入Tensor中连续且重复的元素去重。

    在 `return_inverse=True` 时，返回一个Tensor，包含输入Tensor中的元素在输出Tensor中的索引。

    在 `return_counts=True` 时，返回一个Tensor，表示输出元素在输入中的个数。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **return_inverse** (bool, 可选) - 是否返回输入中每个元素映射到输出中位置的索引。默认值： ``False`` 。
        - **return_counts** (bool, 可选) - 是否返回每个去重元素在输入所在的连续序列的出现次数。默认值： ``False`` 。
        - **dim** (int, 可选) - 维度。如果为 ``None`` ，则对输入进行展平操作。如果指定维度，则必须是int32或int64类型。默认值： ``None`` 。

    返回：
        Tensor或包含Tensor对象的元组（ `output` 、 `inverse_indices` 、 `counts` ）。

        - **output** (Tensor)，`output` 为去重后的输出，与 `input` 具有相同的数据类型。
        - **inverse_indices** (Tensor, 可选)，如果 `return_inverse` 为 ``True`` ，则返回Tensor `inverse_indices` 。若Tensor `inverse_indices` 的shape与 `input` 相同，则表示每个输入中元素映射到输出中位置的索引。
        - **counts** (Tensor, 可选)，如果 `return_counts` 为 ``True`` ，则返回Tensor `counts`。若Tensor `counts` 的shape与 `output` 相同，或当给定dim值时为 `output.shape[dim]` ，则表示每个去重元素在输入中所在的连续序列的计数。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `input` 的数据类型不支持。
        - **TypeError** - `return_inverse` 不是bool。
        - **TypeError** - `return_counts` 不是bool。
        - **TypeError** - `dim` 不是int。
        - **ValueError** - `dim` 不在 :math:`[-ndim, ndim-1]` 范围内。
