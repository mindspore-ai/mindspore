mindspore.Tensor.item
=====================

.. py:method:: mindspore.Tensor.item(index=None)

    获取Tensor中指定索引的元素。

    .. note::
        Tensor.item返回的是Tensor标量，而不是Python标量。

    参数：
        - **index** (Union[None, int, tuple(int)]) - Tensor的索引。默认值：None。

    返回：
        Tensor标量，dtype与原始Tensor的相同。

    异常：
        - **ValueError** - `index` 的长度不等于Tensor的ndim。