mindspore.ops.topk
===================

.. py:function:: mindspore.ops.topk(input_x, k, dim=None, largest=True, sorted=True)

    沿给定维度查找 `k` 个最大或最小元素和对应的索引。

    .. warning::
        - 如果 `sorted` 设置为False，它将使用aicpu运算符，性能可能会降低。

    如果 `input_x` 是一维Tensor，则查找Tensor中 `k` 个最大或最小元素，并将其值和索引输出为Tensor。`values[k]` 是 `input_x` 中 `k` 个最大元素，其索引是 `indices[k]` 。

    对于多维矩阵，计算给定维度中最大或最小的 `k` 个元素，因此：

    .. math::
        values.shape = indices.shape

    如果两个比较的元素相同，则优先返回索引值较小的元素。

    参数：
        - **input_x** (Tensor) - 需计算的输入，数据类型必须为float16、float32或int32。
        - **k** (int) - 指定计算最大或最小元素的数量，必须为常量。
        - **dim** (int, 可选) - 需要排序的维度。默认值：None。
        - **largest** (bool, 可选) - 如果为False，则会返回前k个最小值。默认值：True。
        - **sorted** (bool, 可选) - 如果为True，则获取的元素将按值降序排序。如果为False，则获取的元素将按值升序排序。默认值：True。

    返回：
        由 `values` 和 `indices` 组成的tuple。

        - **values** (Tensor) - 给定维度的每个切片中的 `k` 最大元素或最小元素。
        - **indices** (Tensor) - `k` 最大元素的对应索引。

    异常：
        - **TypeError** - 如果 `sorted` 不是bool。
        - **TypeError** - 如果 `input_x` 不是Tensor。
        - **TypeError** - 如果 `k` 不是int。
        - **TypeError** - 如果 `input_x` 的数据类型不是以下之一：float16、float32或int32。