mindspore.dataset.config.get_multiprocessing_start_method
=========================================================

.. py:function:: mindspore.dataset.config.get_multiprocessing_start_method()

    获取数据预处理子进程启动方式。

    如果 ``set_multiprocessing_start_method`` 没有被调用过，默认返回 ``'fork'`` 。

    返回：
        str，表示数据预处理阶段子进程启动方式。
