mindspore.device_context.ascend.op_tuning.aclnn_cache
======================================================

.. py:function:: mindspore.device_context.ascend.op_tuning.aclnn_cache(enable_global_cache=False, cache_queue_length=10000)

    配置aclnn单算子缓存参数。

    参数：
        - **enable_global_cache** (bool，可选) - 是否开启aclnn全局缓存，此参数只在GRAPH模式下生效。默认值：``False``。
        - **cache_queue_length** (int，可选) - 配置aclnn缓存队列的长度，默认值：``10000``。
