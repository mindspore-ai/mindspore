mindspore.get_offload_context
==============================

.. py:function:: mindspore.get_offload_context()

    获取offload_config配置参数。通过接口mindspore.set_offload_context()进行配置。如果用户未设置，则获取到默认配置。

    返回：
        dict，异构训练offload_config详细配置参数。
