mindspore.get_auto_parallel_context
====================================

.. py:function:: mindspore.get_auto_parallel_context(attr_key)

    根据key获取自动并行的配置。

    参数：
        - **attr_key** (str) - 配置的key。

    返回：
        根据key返回配置的值。

    异常：
        - **ValueError** - 输入key不在自动并行的配置列表中。
