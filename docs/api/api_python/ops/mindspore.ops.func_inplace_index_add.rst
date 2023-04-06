mindspore.ops.inplace_index_add
===============================

.. py:function:: mindspore.ops.inplace_index_add(var, indices, updates, axis)

    逐元素将一个Tensor `updates` 添加到原Tensor `var` 的指定轴和索引处。

    参数：
        - **var** (Parameter) - 被添加的输入Parameter，数据类型为uint8、int8、int16、int32、float16、float32或float64。
        - **indices** (Tensor) - `axis` 指定轴上执行添加操作的索引。是一个1D Tensor，shape为 :math:`(updates.shape[axis],)` ，它的每个值应在范围 :math:`[0, var.shape[axis])` 之内，数据类型为int32。
        - **updates** (Tensor) - 要添加的输入Tensor。必须与 `var` 具有相同的数据类型。 除 `axis` 维度外， `updates` 与 `var` 的shape应一致。
        - **axis** (int) - 要执行添加操作的轴。应该在范围 :math:`[0, len(var.dim))` 内。

    返回：
        Tensor，更新后的结果，其shape和dtype与 `var` 一致。

    异常：
        - **TypeError** - `var` 不是Parameter。
        - **TypeError** - `indices` 或 `updates` 不是Tensor。
        - **ValueError** - `axis` 超出有效范围。
        - **ValueError** - `var` 和 `updates` 的秩不相等。
        - **ValueError** - `indices` shape不是 :math:`(updates.shape[axis],)` 。
        - **ValueError** - `updates` 的shape与 `var` 在除 `axis` 维度外存在不同。

