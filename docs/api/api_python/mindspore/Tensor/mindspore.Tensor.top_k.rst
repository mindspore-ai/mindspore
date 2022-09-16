mindspore.Tensor.top_k
======================

.. py:method:: mindspore.Tensor.top_k(k, sorted=True)

    沿最后一个维度查找 `k` 个最大元素和对应的索引。

    .. warning::
        - 如果 `sorted` 设置为'False'，它将使用aicpu运算符，性能可能会降低。

    `x` 指的当前 Tensor。

    如果 `input_x` 是一维Tensor，则查找Tensor中 `k` 个最大元素，并将其值和索引输出为Tensor。因此， `values[k]` 是 `input_x` 中 `k` 个最大元素，其索引是 `indices[k]` 。

    对于多维矩阵，计算每行中最大的 `k` 个元素（沿最后一个维度的相应向量），因此：

    .. math::
        values.shape = indices.shape = input\_x.shape[:-1] + [k].

    如果两个比较的元素相同，则优先返回索引值较小的元素。

    参数：
        - **k** (int) - 指定计算最大元素的数量，需要是常量。
        - **sorted** (bool, 可选) - 如果为True，则获取的元素将按值降序排序。默认值：True。

    返回：
        2个Tensor组成的tuple， `values` 和 `indices` 。

        - **values** (Tensor) - 最后一个维度的每个切片中的 `k` 最大元素。
        - **indices** (Tensor) - `k` 最大元素的对应索引。

    异常：
        - **TypeError** - 如果 `k` 不是int。
        - **TypeError** - 如果 `sorted` 不是bool。