mindspore.Tensor.topk
======================

.. py:method:: mindspore.Tensor.topk(k, dim=-1, largest=True, sorted=True)

    沿给定维度查找 `k` 个最大或最小的元素，并返回其值和对应的索引。

    .. warning::
        由于不同平台上的内存布局和遍历方法不同：当 `sorted` 设置为False时，计算结果的显示顺序可能不一致。

    如果 `self` 是一维Tensor，则查找Tensor中 `k` 个最大或最小元素，并且将其值和索引输出为Tensor。 `values[k]` 是 `self` 中 `k` 个最大元素，同时其索引值为 `indices[k]` 。

    对于多维矩阵，计算给定维度中最大或最小的 `k` 个元素，因此：

    .. math::

        values.shape = indices.shape

    如果两个比较的元素相同，则优先返回索引值较小的元素。

    参数：
        - **k** (int) - 指定计算最大或最小元素的数量，必须为常量。
        - **dim** (int, 可选) - 需要排序的维度。默认值: ``-1`` 。
        - **largest** (bool, 可选) - 如果设置为 ``False`` ，则会返回 `k` 个最小元素。默认值： ``True`` 。
        - **sorted** (bool, 可选) - 如果设置为 ``True`` ，则获取的元素将按值降序排序；如果设置为 ``False``，则不对获取的元素进行排序，默认值： ``True`` 。

    返回：
        由 `values` 和 `indices` 组成的tuple。
        - **values** (Tensor) - 给定维度的每个切片中的 `k` 最大元素或最小元素。
        - **indices** (Tensor) - `k` 最大元素的对应索引。

    异常：
        - **TypeError** - 如果 `sorted` 不是bool。
        - **TypeError** - 如果 `k` 不是int。

    .. py:method:: mindspore.Tensor.topk(k, dim=None, largest=True, sorted=True)
        :noindex:

    详情请参考 :func:`mindspore.ops.topk` 。
