mindspore.runtime.PluggableAllocator
====================================

.. py:class:: mindspore.runtime.PluggableAllocator(path_to_so_file, alloc_fn_name, free_fn_name)

    通过ctypes接收一个.so文件，并动态加载其中的alloc和free函数。
    该功能需要与 :class:`mindspore.runtime.MemPool` 和 :func:`mindspore.runtime.use_mem_pool` 配合使用，以接管MindSpore内存池中的内存分配和释放操作。

    .. warning::
        当前仅支持在Unix类操作系统上使用。

    参数：
        - **path_to_so_file** (str) - 文件系统中包含分配器函数的 `.so` 文件的路径。
        - **alloc_fn_name** (str) - so文件中执行内存分配的函数名称。函数签名必须为：
          `void* alloc_fn(size_t size, int device, aclrtStream stream);` 。
        - **free_fn_name** (str) - so文件中执行内存释放的函数名称。函数签名必须为：
          `void free_fn(void* ptr, size_t size, aclrtStream stream);` 。
