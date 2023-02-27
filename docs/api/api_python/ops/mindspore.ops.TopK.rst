mindspore.ops.TopK
===================

.. py:class:: mindspore.ops.TopK(sorted=True)

    沿最后一个维度查找 `k` 个最大元素和对应的索引。

    .. warning::
        - 如果 `sorted` 设置为False，它将使用aicpu运算符，性能可能会降低，另外，由于在不同平台上存在内存排布以及遍历方式不同等问题，`sorted` 设置为False时计算结果的显示顺序可能会出现不一致的情况。

    如果 `input_x` 是一维Tensor，则查找Tensor中 `k` 个最大元素，并将其值和索引输出为Tensor。`values[k]` 是 `input_x` 中 `k` 个最大元素，其索引是 `indices[k]` 。

    对于多维矩阵，计算每行中最大的 `k` 个元素（沿最后一个维度的相应向量），因此：

    .. math::
        values.shape = indices.shape = input.shape[:-1] + [k].

    如果两个比较的元素相同，则优先返回索引值较小的元素。

    参数：
        - **sorted** (bool, 可选) - 如果为True，则获取的元素将按值降序排序。如果为False，则不对获取的元素进行排序。默认值：True。

    输入：
        - **input_x** (Tensor) - 需计算的输入，CPU推理数据类型必须为float16、float32或int32；GPU推理数据类型必须为float16或float32。
        - **k** (int) - 指定计算最大元素的数量，必须为常量。

    输出：
        由 `values` 和 `indices` 组成的tuple。

        - **values** (Tensor) - 最后一个维度的每个切片中的 `k` 最大元素。
        - **indices** (Tensor) - `k` 最大元素的对应索引。

    异常：
        - **TypeError** - 如果 `sorted` 不是bool。
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **TypeError** - 如果 `k` 不是int。
        - **TypeError** - 如果 `input_x` 的数据类型不是以下之一：float16、float32或int32。