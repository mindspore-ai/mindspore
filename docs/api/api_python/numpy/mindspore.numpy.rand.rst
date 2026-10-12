mindspore.numpy.rand
=================================

.. py:function:: mindspore.numpy.rand(*shape, dtype=mstype.float32)

    返回一个给定shape和类型的新Tensor，其中所有元素以区间 :math:`[0,1)` 上均匀分布的随机数来填充。

    参数：
        - **shape** (Union[int, tuple(int), list(int)]) - 指定新Tensor的shape，例如 :math:`(2,3)` 或 :math:`2` 。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定Tensor的数据类型。必须为float类型。默认值： ``mstype.float32`` 。

    返回：
        Tensor，给定shape和类型，其中所有元素都为区间 :math:`[0,1)` 上均匀分布的随机数。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 ``dtype`` 不是float类型。
