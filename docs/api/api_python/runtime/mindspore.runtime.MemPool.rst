mindspore.runtime.MemPool
==========================

.. py:class:: mindspore.runtime.MemPool(allocator)

    MemPool封装了一个 :class:`mindspore.runtime.PluggableAllocator` ，并将其传递给 :func:`mindspore.runtime.use_mem_pool` 使用。

    参数：
        - **allocator** (mindspore.runtime.PluggableAllocator) - 一个 :class:`mindspore.runtime.PluggableAllocator` 对象，用于定义内存池中内存的分配和释放方式。
