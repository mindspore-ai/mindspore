mindspore.runtime.use_mem_pool
===============================

.. py:function:: mindspore.runtime.use_mem_pool(pool)

    将内存分配和释放操作路由到指定内存池的上下文管理器。

    .. note::
        - 该上下文管理器仅会使当前线程的内存分配操作路由到指定内存池。
        - 若在上下文管理器内部创建新线程，该线程的内存分配将不会路由到指定内存池。
        - 只有在上下文管理器内部分配Device内存，才能将分配操作路由到指定内存池。
        - 仅Atlas A2训练系列产品支持该接口。

    参数：
        - **pool** (mindspore.runtime.MemPool) - 封装了PluggableAllocator的MemPool对象。
