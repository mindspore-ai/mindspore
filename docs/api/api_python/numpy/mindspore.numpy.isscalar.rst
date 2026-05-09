mindspore.numpy.isscalar
=================================

.. py:function:: mindspore.numpy.isscalar(element)

    如果元素的类型是标量类型，则返回True。

    .. note::
        仅支持MindSpore解析器识别的对象类型，包括在MindSpore范围内定义的对象、类型、方法和函数。不支持其他内置类型。

    参数：
        - **element** (any) - 输入参数，可以是任意类型和shape。

    返回：
        Boolean，如果 `element` 是标量类型，则返回True，否则返回False。

    异常：
        - **TypeError** - 如果 `element` 的类型不受MindSpore解析器支持。