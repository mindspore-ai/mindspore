mindspore.get_context
======================

.. py:function:: mindspore.get_context(attr_key)

    根据输入key获取context中的属性值，此接口将在后续版本中废弃，请使用接口 :func:`mindspore.get_current_device` 替代。

    若该属性未设置，则返回其默认值。

    参数：
        - **attr_key** (str) - 属性的key。

    返回：
        Object，表示给定属性key的值。

    异常：
        - **ValueError** - 输入key不是context中的属性。
