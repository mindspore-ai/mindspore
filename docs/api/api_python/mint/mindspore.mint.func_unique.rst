mindspore.mint.unique
=====================

.. py:function:: mindspore.mint.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)

    对输入tensor中元素去重。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **sorted** (bool，可选) - 是否对返回值进行升序排序。默认 ``True`` 。
        - **return_inverse** (bool，可选) - 是否额外返回一个tensor，表示 `input` 在 `output` 上对应的index。默认 ``False`` 。
        - **return_counts** (bool，可选) - 是否额外返回一个tensor，表示输出 `output` 中的每个元素在 `input` 中的数量。默认 ``False`` 。
        - **dim** (int，可选) - 指定计算维度。

    返回：
        一个tensor，或者几个tensor的集合：

        - **output** (Tensor) - `input` 中去重后的元素。
        - **inverse_indices** (Tensor) - 当 `return_inverse=True` 时返回，表示 `input` 中的每个元素在 `output` 上对应的位置索引。当 `dim=None` 时，shape和 `input` 一样；当 `dim` 有值的时候，shape是input.shape[dim]。
        - **counts** (Tensor) - 当 `return_counts=True` 时返回，表示 `output` 中每个元素在 `input` 中的数量。当 `dim=None` 时，shape和 `output` 一样；当 `dim` 有值的时候，shape是output.shape[dim]。
