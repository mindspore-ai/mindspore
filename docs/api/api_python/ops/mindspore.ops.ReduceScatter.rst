mindspore.ops.ReduceScatter
===========================

.. py:class:: mindspore.ops.ReduceScatter(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约并且分发指定通信组中的张量。

    .. note::
        在集合的所有过程中，Tensor必须具有相同的shape和格式。
        在运行以下示例之前，用户需要预设环境变量。请参考
        `MindSpore官方网站 <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.ops.html#%E9%80%9A%E4%BF%A1%E7%AE%97%E5%AD%90>`_。

    参数：
        - **op** (str) - 指定用于元素的规约操作，如SUM和MAX。默认值：ReduceOp.SUM。
        - **group** (str) - 要处理的通信组。默认值："GlobalComm.WORLD_COMM_group"。

    异常：
        - **TypeError** - 如果 `op` 和 `group` 不是字符串。
        - **ValueError** - 如果输入的第一个维度不能被rank size整除。rank size是指通信组通信的卡数。
