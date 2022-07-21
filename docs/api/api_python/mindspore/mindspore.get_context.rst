mindspore.get_context
======================

.. py:function:: mindspore.get_context(attr_key)

    根据输入key获取context中的属性值。如果该key没有设置，则会获取它们这些的默认值。

    参数：
        - **attr_key** (str) - 属性的key。

    返回：
        Object，表示给定属性key的值。

    异常：
        - **ValueError** - 输入key不是context中的属性。
