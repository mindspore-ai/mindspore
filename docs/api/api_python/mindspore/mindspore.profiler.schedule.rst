mindspore.profiler.schedule
===========================

.. py:class:: mindspore.profiler.schedule(*, wait, active, warmup=0, repeat=0, skip_first=0)

    该类用于获取每一步的操作。

    调度如下：

    .. code-block::

        (NONE)        (NONE)          (NONE)       (WARM_UP)       (RECORD)      (RECORD)     (RECORD_AND_SAVE)    None
        START------->skip_first------->wait-------->warmup-------->active........active.........active----------->stop
                                      |                                                             |
                                      |                           repeat_1                          |
                                      ---------------------------------------------------------------

    Profiler将跳过前 ``skip_first`` 步，然后等待 ``wait`` 步，接着在接下来的 ``warmup`` 步中进行预热，然后在接下来的 ``active`` 步中进行活动记录，然后从 ``wait`` 步开始重复循环。可选的循环次数由 ``repeat`` 参数指定， ``repeat`` 值为0表示循环将继续直到分析完成。


    关键字参数：
        - **wait** (int) - 预热阶段等待的步数，必须大于等于0。如果外部不设置 `wait` 参数，会在初始化schedule类时，设置为 ``0``。
        - **active** (int) - 活动阶段执行的步数，必须大于等于1。如果外部不设置 `active` 参数，会在初始化schedule类时，设置为 ``1``。
        - **warmup** (int, 可选) - 预热阶段执行的步数，必须大于等于0。默认值： ``0`` 。
        - **repeat** (int, 可选) - 重复次数，必须大于等于0。如果 `repeat` 设置为 ``0``，Profiler会根据模型训练次数来确定repeat值，例如总训练步数为100，wait+active+warmup=10，skip_first=10，则repeat=(100-10)/10=9，表示重复执行9次，但此时会多生成一个采集不完整的性能数据，最后一个step的数据用户无需关注，为异常数据。建议配置大于0的整数。当使用集群分析工具或MindStudio Insight查看时，建议配置为1；若设置大于1，则需要将采集的性能数据文件夹分为repeat等份，放到不同文件夹下重新解析，分类方式按照文件夹名称中的时间戳先后。默认值： ``0`` 。
        - **skip_first** (int, 可选) - 跳过第一个步数。默认值： ``0`` 。

    异常：
        - **ValueError** - 参数step必须大于等于0。

    .. py:method:: to_dict()

        将schedule类转换为一个字典。

        返回：
            字典，schedule类的参数与对应的值。
