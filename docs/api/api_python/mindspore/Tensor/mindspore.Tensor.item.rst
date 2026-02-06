mindspore.Tensor.item
=====================

.. py:method:: mindspore.Tensor.item()

    返回只包含一个元素的Tensor的元素值。

    返回：
        标量，类型由Tensor的dtype决定。

    异常：
        - **ValueError** - Tensor的元素个数不止一个。
        - **TypeError** - 调用 `item` 接口的Tensor的元素数据类型不被支持。
