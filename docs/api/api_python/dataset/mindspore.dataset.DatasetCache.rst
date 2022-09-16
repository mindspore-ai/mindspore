mindspore.dataset.DatasetCache
==============================

.. py:class:: mindspore.dataset.DatasetCache(session_id, size=0, spilling=False, hostname=None, port=None, num_connections=None, prefetch_size=None)

    创建数据缓存客户端实例。

    关于单节点数据缓存的使用，请参阅 `单节点数据缓存教程 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/dataset/cache.html>`_ 。

    参数：
        - **session_id** (int) - 当前数据缓存客户端的会话ID，用户在命令行开启缓存服务端后可通过 `cache_admin -g` 获取。
        - **size** (int, 可选) - 设置数据缓存服务可用的内存大小。默认值：0，表示内存使用没有限制。
        - **spilling** (bool, 可选) - 如果共享内存不足，是否将溢出部分缓存到磁盘。默认值：False。
        - **hostname** (str, 可选) - 数据缓存服务客户端的主机IP。默认值：None，表示使用默认主机IP 127.0.0.1。
        - **port** (int, 可选) - 指定连接到数据缓存服务端的端口号。默认值：None，表示端口为50052。
        - **num_connections** (int, 可选) - TCP/IP连接数量。默认值：None，表示连接数量为12。
        - **prefetch_size** (int, 可选) - 指定缓存队列大小，使用缓存功能算子时，将直接从缓存队列中获取数据。默认值：None，表示缓存队列大小为20。

    .. py:method:: get_stat()

        获取缓存实例的统计信息。在数据管道结束后，可获取三类统计信息，包括平均缓存命中数（avg_cache_sz），内存中的缓存数（num_mem_cached）和磁盘中的缓存数（num_disk_cached）。
