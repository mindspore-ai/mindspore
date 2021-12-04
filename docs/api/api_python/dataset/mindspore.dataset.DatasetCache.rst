mindspore.dataset.DatasetCache
==============================

.. py:class:: mindspore.dataset.DatasetCache(session_id, size=0, spilling=False, hostname=None, port=None, num_connections=None, prefetch_size=None)

    创建数据缓存客户端实例。

    有关详细信息，请查看 `教程 <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_cache.html>`_ 、
    `编程指南 <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cache.html>`_。

    **参数：**

    - **session_id** (int) - 当前数据缓存客户端的会话ID，用户在命令行开启缓存服务端后可通过 `cache_admin -g` 获取。
    - **size** (int, optional) - 设置数据缓存服务可用的内存大小（默认为0，即内存使用没有上限。注意，这可能会产生计算机内存不足的风险）。
    - **spilling** (bool, optional) - 如果共享内存不足，是否将溢出部分缓存到磁盘（默认为False）。
    - **hostname** (str, optional) - 数据缓存服务客户端的主机IP（默认为None，使用默认主机名127.0.0.1）。
    - **port** (int, optional) - 指定连接到数据缓存服务端的端口号（默认为None，使用端口50052）。
    - **num_connections** (int, optional) - TCP/IP连接数量（默认为None，使用默认值12）。
    - **prefetch_size** (int, optional) - 指定缓存队列大小，使用缓存功能算子时，将直接从缓存队列中获取数据（默认为None，使用默认值20）。

    **样例：**

    >>> import mindspore.dataset as ds
    >>>
    >>> # 创建数据缓存客户端实例，其中 `session_id` 由命令 `cache_admin -g` 生成
    >>> some_cache = ds.DatasetCache(session_id=session_id, size=0)
    >>>
    >>> dataset_dir = "path/to/imagefolder_directory"
    >>> ds1 = ds.ImageFolderDataset(dataset_dir, cache=some_cache)

    .. py:method:: get_stat()

        获取缓存实例的统计信息。