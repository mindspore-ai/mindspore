mindspore.numpy.ones
=================================

.. py:function:: mindspore.numpy.ones(shape, dtype=mstype.float32)

    返回一个给定shape和类型的新Tensor，其中所有元素用1来填充。

    参数：
        - **shape** (Union[int, tuple, list]) - 指定Tensor的shape。
        - **dtype** (Union[mindspore.dtype, str], 可选) - 指定的Tensor `dtype` 。默认值： ``mstype.float32`` 。

    返回：
        Tensor，给定 `shape` 和 `dtype` ，其中所有元素都为1。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 `shape` 的元素项值 :math:`<0` 。
