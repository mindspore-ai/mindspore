mindspore.profiler.profiler.analyse
===================================

.. py:function:: mindspore.profiler.profiler.analyse(profiler_path, max_process_number=os.cpu_count() // 2, pretty=False, step_list=None, data_simplification=True)

    离线分析训练的性能数据，性能数据采集结束后调用。

    参数：
        - **profiler_path** (str) - 需要进行离线分析的profiling数据路径，指定到 ``*_ascend_ms`` 上层目录。
        - **max_process_number** (int, 可选) - 最大进程数，默认值：``os.cpu_count() // 2`` 。
        - **pretty** (bool, 可选) - 对json文件进行格式化处理。默认值：``False``，即不进行格式化。
        - **step_list** (list, 可选) - 只解析指定step的性能数据，指定的step必须是连续的整数。仅支持GRAPH模式下CallBack方式采集，只能切分CANN层及以下信息。默认值：``None``，即进行全解析。
        - **data_simplification** (bool, 可选) - 是否开启数据精简开关功能。默认值：``True``，即开启数据精简。
