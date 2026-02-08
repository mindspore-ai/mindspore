mindspore.Tensor.unique
=======================

.. py:method:: mindspore.Tensor.unique(sorted=True, return_inverse=False, return_counts=False, dim=None)

    对 `self` 中的元素去重。

    在 `return_inverse=True` 时，会返回一个索引Tensor，包含 `self` 中的元素在输出Tensor中的索引。

    在 `return_counts=True` 时，会返回一个Tensor，表示输出元素在 `self` 中的个数。

    参数：
        - **sorted** (bool，可选) - 输出是否需要进行升序排序。默认值： ``True`` 。
        - **return_inverse** (bool，可选) - 是否输出 `self` 在 `output` 上对应的index。默认值： ``False`` 。
        - **return_counts** (bool，可选) - 是否输出 `output` 中元素的数量。默认值： ``False`` 。
        - **dim** (int，可选) - 做去重操作的维度。当 `dim` 为 ``None`` 时，对展开的 `self` 做去重操作，否则，将给定维度的Tensor视为一个元素做去重操作。默认值： ``None`` 。

    返回：
        输出为一个Tensor，或者以下一个或几个Tensor的集合：（ `output` ， `inverse_indices` ， `counts` ）。

        - **output** (Tensor) - 与 `self` 数据类型相同的Tensor，包含 `self` 中去重后的元素。
        - **inverse_indices** (Tensor，可选) - 当 `return_inverse=True` 时返回，表示 `self` 中的元素在输出Tensor中的索引。当 `dim=None` 时，shape和 `self` 相同；否则，shape为self.shape[dim]。
        - **counts** (Tensor，可选) - 当 `return_counts=True` 时返回，表示输出Tensor中元素在 `self` 中的数量。当 `dim=None` 时，shape和 `output` 相同；否则，shape是output.shape[dim]。
