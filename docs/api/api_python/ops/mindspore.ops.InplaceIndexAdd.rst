mindspore.ops.InplaceIndexAdd
=============================

.. py:class:: mindspore.ops.InplaceIndexAdd(axis)

    逐元素将一个Tensor `updates` 添加到原Tensor `var` 的指定轴和索引处。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.inplace_index_add`。

    参数：
        - **axis** (int) - 要执行添加操作的轴。应该在范围 :math:`[0, len(var.dim))` 内。

    输入：
        - **var** (Parameter) - 被添加的输入Parameter，数据类型为uint8、int8、int16、int32、float16、float32或float64。
        - **indices** (Tensor) - `axis` 指定轴上执行添加操作的索引。是一个1D Tensor，shape为 :math:`(updates.shape[axis],)` ，它的每个值应在范围 :math:`[0, var.shape[axis])` 之内，数据类型为int32。
        - **updates** (Tensor) - 要添加的输入Tensor。必须与 `var` 具有相同的数据类型。 除 `axis` 维度外， `updates` 与 `var` 的shape应一致。

    输出：
        Tensor，更新后的结果，其shape和dtype与 `var` 一致。
