mindspore.numpy.argwhere
=================================

.. py:function:: mindspore.numpy.argwhere(a)

    返回Tensor中非零元素的索引，并按元素分组。

    参数：
        - **a** (Union[list, tuple, Tensor]) - 输入的Tensor。

    返回：
        Tensor，非零元素的索引。索引按元素分组。Tensor的shape为 :math:`(N, a.ndim)` 其中 :math:`N` 为非零元素个数。

    异常：
        - **TypeError** - 如果 ``a`` 不是类似数组的对象。
        - **ValueError** - 如果 ``a`` 的维数为0。