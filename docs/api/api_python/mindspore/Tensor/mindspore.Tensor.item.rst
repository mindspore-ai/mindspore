mindspore.Tensor.item
=====================

.. py:method:: mindspore.Tensor.item(index=None)

    获取Tensor中指定索引的元素。

    参数：
        - **index** (Union[None, int, tuple(int)]) - Tensor的索引。默认值： ``None`` 。

    返回：
        标量，类型由Tensor的dtype决定。

    异常：
        - **ValueError** - `index` 的长度不等于Tensor的ndim。
