mindspore.profiler
========================

profiler模块简介。

本模块提供Python API，用于启用MindSpore神经网络性能数据的分析。
用户可以通过 ``import mindspore.profiler.Profiler`` 并初始化Profiler对象以开始分析，并使用 `Profiler.analyse()` 停止收集和分析。
用户可通过Mindinsight工具可视化分析结果。
目前，Profiler支持AICore算子、AICpu算子、HostCpu算子、内存、设备通信、集群等数据的分析。

.. py:class:: mindspore.profiler.Profiler(**kwargs)

    性能采集API。

    此API能够让MindSpore用户采集神经网络的性能。
    Profiler支持Ascend和GPU，两者的使用方式相同。

    **参数：**

    - **output_path** (str) – 表示输出数据的路径。
    - **optypes_not_deal** (str) – （仅限Ascend）该参数已弃用，该功能已不再支持。
    - **ascend_job_id** (str) – （仅限Ascend）该参数已弃用，该功能已不再支持。
    - **profile_communication** (bool) – （仅限Ascend）表示是否在多设备训练中收集通信性能数据。当值为True时，收集这些数据。默认值为False。在单台设备训练中，该参数的设置无效。
    - **profile_memory** (bool) – （仅限Ascend）表示是否收集Tensor内存数据。当值为True时，收集这些数据。默认值为False。
    - **start_profile** (bool) – 该参数控制是否在Profiler初始化的时候开启采集数据。默认值为True。

    **异常：**

    - **RuntimeError** – 当CANN的版本与MindSpore版本不匹配时，生成的ascend_job_id目录结构MindSpore无法解析。

    .. py:method:: analyse()

        收集和分析训练后或训练期间调用的性能数据。样例如上所示。

    .. py:method:: profile(network,profile_option)

        获取训练网络中可训练参数的数量。

        **参数：**

        - **network** (Cell) - 表示训练网络。
        - **profile_option** (ProfileOption) - 该参数已弃用，该功能已不再支持。

        **返回：**

        dict，其中key为选项名称，value为选项结果。

    .. py:method:: start()

        开启Profiler数据采集，可以按条件开启Profiler。

        **异常：**

        - **RuntimeError** – profiler已经开启。
        - **RuntimeError** – 停止Minddata采集后，不支持重复开启。
        - **RuntimeError** – 如果start_profile参数设置为True。

    .. py:method:: stop()

        停止Profiler，可以按条件停止Profiler。

        **异常：**

        - **RuntimeError** – profiler没有开启。

.. py:class:: mindspore.profiler.ProfileOption

    这个类已经弃用，该功能已不再支持。