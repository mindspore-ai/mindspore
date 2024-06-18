mindspore.mint.unique
=====================

.. py:function:: mindspore.mint.unique(input, sorted=True, return_inverse=False, return_counts=False, dim=None)

    对输入Tensor中元素去重。

    在 `return_inverse=True` 时，会返回一个索引Tensor，包含输入Tensor中的元素在输出Tensor中的索引；
    在 `return_counts=True` 时，会返回一个Tensor，表示输出元素在输入中的个数。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **sorted** (bool) - 输出是否需要进行升序排序。默认值： ``True`` 。
        - **return_inverse** (bool) - 是否输出 `input` 在 `output` 上对应的index。默认值： ``False`` 。
        - **return_counts** (bool) - 是否输出 `output` 中元素的数量。默认值： ``False`` 。
        - **dim** (int) - 做去重操作的维度，当设置为 ``None`` 的时候，对展开的输入做去重操作, 否则，将给定维度的Tensor视为一个元素去做去重操作。默认值：``None`` 。

    返回：
        输出为一个Tensor，或者以下一个或几个Tensor的集合：（`output`，`inverse_indeices`，`counts`）

        - **output** (Tensor) - 与 `input` 数据类型相同的Tensor，包含 `input` 中去重后的元素。
        - **inverse_indeices** (Tensor) - 当 `return_inverse=True` 时返回，表示输入Tensor中的元素在输出Tensor中的索引。当 `dim=None` 时，shape和 `input` 一样；当 `dim` 有值的时候，shape是input.shape[dim]。
        - **counts** (Tensor) - 当 `return_counts=True` 时返回，表示输出Tensor中元素在输入Tensor中的数量。当 `dim=None` 时，shape和 `output` 一样；当 `dim` 有值的时候，shape是output.shape[dim]。

    异常：
        - **TypeError** - `input` 不是Tensor。
