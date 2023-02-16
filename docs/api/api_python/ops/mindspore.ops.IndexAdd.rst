mindspore.ops.IndexAdd
======================

.. py:class:: mindspore.ops.IndexAdd(axis, use_lock=True, check_index_bound=True)

    将Tensor `y` 加到Tensor `x` 的指定 `axis` 和 `indices` 。 `axis` 取值范围为[-len(x.dim),  len(x.dim) - 1]， `indices` 取值范围为[0, len(x[axis]) - 1]。

    参数：
        - **axis** (int) - 进行索引的axis。
        - **use_lock** (bool) - 是否对参数更新加锁保护。默认值：True。
        - **check_index_bound** (bool) - 如果为True将对索引进行边界检查。默认值：True。

    输入：
        - **x** (Parameter) - 要添加到的输入参数。
        - **indices** (Tensor) - 沿 `axis` 在指定 `indices` 位置进行加法运算。数据类型支持int32。`indices` 必须为一维且与 `y` 在 `axis` 维度的尺寸相同。 `indices` 取值范围应为[0, b)，其中b为 `x` 在 `axis` 维度的尺寸。
        - **y** (Tensor) - 被添加到 `x` 的输入Tensor。必须与 `x` 的数据类型相同。除 `axis` 之外的维度shape必须与 `x` 的shape相同。

    输出：
        Tensor，与 `x` 的shape和数据类型相同。

    异常：
        - **TypeError** - `x` 不是Parameter。
        - **TypeError** - `indices` 或 `y` 不是Tensor。
        - **ValueError** - axis 超过了 `x` 的秩。
        - **ValueError** - `x` 与 `y` 的秩不相同。
        - **ValueError** - `indices` 不是一维或与 `y[axis]` 的尺寸不同。
        - **ValueError** - `y` 的shape与除 `axis` 之外的维度的 `x` 的shape不同。
