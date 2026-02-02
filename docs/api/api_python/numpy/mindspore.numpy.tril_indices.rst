mindspore.numpy.tril_indices
=================================

.. py:function:: mindspore.numpy.tril_indices(n, k=0, m=None)

    返回shape为 `(n, m)` 数组的下三角的索引。

    参数：
        - **n** (int) - 指定返回索引适用的数组大小。
        - **k** (int, 可选) - 对角线的偏移量。默认值： ``0`` 。
        - **m** (int, 可选) - 指定返回索引适用的数组的列维度，默认 ``None`` ，当为 ``None`` 时等于 `n` 。

    返回：
        下三角形的索引。返回的Tuple包含两个Tensor，每个Tensor分别对应其在一个维度上的索引。

    异常：
        - **TypeError** - 如果 `n` 、 `k` 、 `m` 不是数字。