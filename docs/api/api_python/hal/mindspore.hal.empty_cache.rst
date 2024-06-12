mindspore.hal.empty_cache
=========================

.. py:function:: mindspore.hal.empty_cache()

    清理内存池中的内存碎片，优化内存排布。

    .. note::
        - 目前MindSpore内存池没有清空内存碎片的功能，该接口预留但实现为空方法并以日志方式提示。
