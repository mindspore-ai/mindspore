mindspore.numpy.zeros
=================================

.. py:function:: mindspore.numpy.zeros(shape, dtype=mstype.float32)

    返回一个给定shape和类型的新Tensor，其中所有元素以0来填充。

    参数：
        - **shape** (Union[int, tuple, list]) - 指定的新Tensor的shape。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的新的Tensor数据类型。默认值： ``mstype.float32`` 。

    返回：
        Tensor，给定 `shape` 和 `dtype` ，其中所有元素都为0。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 `shape` 的元素项值 :math:`<0` 。