mindspore.numpy.diag_indices
=================================

.. py:function:: mindspore.numpy.diag_indices(n, ndim=2)

    返回一个可以访问数组的主对角线的索引数组。
    该函数返回一个索引的元组，索引指向数组a的主对角线，数组a满足 ``a.ndim >= 2`` 与shape为 `(n, n, …, n)` 。如果满足 ``a.ndim = 2`` ，该接口与diagonal功能一致；如果 ``a.ndim > 2`` ，则返回访问 ``a[i, i, ..., i]`` 的索引集合，其中 ``i = [0..n-1]`` 。

    参数：
        - **n** (int) - 可使用返回索引的数组沿每个维度的大小。
        - **ndim** (int, 可选) - 维数。默认值： ``2`` 。

    返回：
        元素为Tensor的Tuple。

    异常：
        - **TypeError** - 如果没有按照上述内容给定的数据类型输入参数。