mindspore.numpy.ones_like
=================================

.. py:function:: mindspore.numpy.ones_like(a, dtype=None, shape=None)

    返回一个与给定数组 `a` 具有相同shape和类型的Tensor，其中所有元素用1来填充。

    .. note::
        输入数组在同一维度上必须具有相同的size。如果 `a` 不是Tensor，且未设置 `dtype` ，那么默认情况下 `dtype` 是float32。

    参数：
        - **a** (Union[Tensor, list, tuple]) - 原数组，返回数组的shape和数据类型与 `a` 相同。
        - **dtype** (mindspore.dtype, 可选) - 覆盖结果的数据类型。默认值： ``None`` 。
        - **shape** (int, ints的序列, 可选) - 覆盖结果的shape。默认值： ``None`` 。

    返回：
        Tensor，与给定 `a` 的shape和数据类型相同，其中所有元素都为1。

    异常：
        - **ValueError** - 如果 `a` 不是Tensor、List或Tuple。
