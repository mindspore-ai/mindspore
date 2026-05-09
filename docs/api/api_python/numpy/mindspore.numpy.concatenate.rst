mindspore.numpy.concatenate
=================================

.. py:function:: mindspore.numpy.concatenate(arrays, axis=0)

    沿现有轴连接一系列Tensor。

    .. note::
        为了匹配Numpy的行为，当 :math:`axis >= 32` 时不会引发值错误， `axis` 将被视为 ``None`` 。

    参数：
        - **arrays** (Union[Tensor, tuple(Tensor), list(Tensor)]) - 要连接的一个Tensor或Tensor列表。
        - **axis** (Union[None, int], 可选) - 沿该轴连接Tensor。如果 `axis` 为 ``None`` ，则在使用前会先将Tensor展平。默认值： ``0`` 。

    返回：
        从一个Tensor或Tensor列表中连接后的Tensor。

    异常：
        - **TypeError** - 如果输入参数非上述给定的类型。
        - **ValueError** - 如果 `axis` 不在范围 :math:`[-ndim, ndim-1]` 内，且小于32。