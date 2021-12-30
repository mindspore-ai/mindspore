mindspore.dataset.DatasetCache
==============================

.. py:class:: mindspore.dataset.DatasetCache(session_id, size=0, spilling=False, hostname=None, port=None, num_connections=None, prefetch_size=None)

    创建数据缓存客户端实例。

    关于单节点数据缓存的使用，请参阅 `单节点数据缓存教程 <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_cache.html>`_ 、
    `单节点数据缓存编程指南 <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cache.html>`_。

    **参数：**

    - **session_id** (int) - 当前数据缓存客户端的会话ID，用户在命令行开启缓存服务端后可通过 `cache_admin -g` 获取。
    - **size** (int, optional) - 设置数据缓存服务可用的内存大小。默认值：0，表示内存使用没有限制。
    - **spilling** (bool, optional) - 如果共享内存不足，是否将溢出部分缓存到磁盘。默认值：False。
    - **hostname** (str, optional) - 数据缓存服务客户端的主机IP。默认值：None，表示使用默认主机IP 127.0.0.1。
    - **port** (int, optional) - 指定连接到数据缓存服务端的端口号。默认值：None，表示端口为50052。
    - **num_connections** (int, optional) - TCP/IP连接数量。默认值：None，表示连接数量为12。
    - **prefetch_size** (int, optional) - 指定缓存队列大小，使用缓存功能算子时，将直接从缓存队列中获取数据。默认值：None，表示缓存队列大小为20。

    **样例：**

    >>> import mindspore.dataset as ds
    >>>
    >>> # 创建数据缓存客户端实例，其中 `session_id` 由命令 `cache_admin -g` 生成
    >>> some_cache = ds.DatasetCache(session_id=session_id, size=0)
    >>>
    >>> dataset_dir = "path/to/imagefolder_directory"
    >>> ds1 = ds.ImageFolderDataset(dataset_dir, cache=some_cache)

    .. py:method:: get_stat()

        获取缓存实例的统计信息。在数据管道结束后，可获取三类统计信息，包括平均缓存命中数（avg_cache_sz），内存中的缓存数（num_mem_cached）和磁盘中的缓存数（num_disk_cached）。

        **样例：**

        >>> import mindspore.dataset as ds
        >>>
        >>> # 创建数据缓存客户端实例，其中 `session_id` 由命令 `cache_admin -g` 生成
        >>> some_cache = ds.DatasetCache(session_id=session_id, size=0)
        >>>
        >>> dataset_dir = "path/to/imagefolder_directory"
        >>> ds1 = ds.ImageFolderDataset(dataset_dir, cache=some_cache)
        >>> for _ in ds1.create_dict_iterator(num_epochs=1):
        ...     pass
        >>> # 数据管道执行结束之后，才能获取cache的统计信息
        >>> stat = some_cache.get_stat()
        >>> # 获取平均缓存命中数（avg_cache_sz）
        >>> cache_sz = stat.avg_cache_sz
        >>> # 获取内存中的缓存数（num_mem_cached）
        >>> num_mem_cached = stat.num_mem_cached
        >>> # 获取磁盘中的缓存数（num_disk_cached）
        >>> num_dick_cached = stat.num_disk_cached