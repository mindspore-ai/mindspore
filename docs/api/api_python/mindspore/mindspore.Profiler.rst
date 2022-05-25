mindspore.profiler
========================

.. py:class:: mindspore.Profiler(**kwargs)

    MindSpore用户能够通过该类对神经网络的性能进行采集。可以通过导入 `mindspore.Profiler` 然后初始化Profiler对象以开始分析，使用 `Profiler.analyse()` 停止收集和分析。可通过Mindinsight工具可视化分析结果。目前，Profiler支持AICORE算子、AICPU算子、HostCPU算子、内存、设备通信、集群等数据的分析。

    **参数：**
    
    - **output_path** (str, 可选) – 表示输出数据的路径。默认值："./data"。
    - **profile_communication** (bool, 可选) – （仅限Ascend）表示是否在多设备训练中收集通信性能数据。当值为True时，收集这些数据。在单台设备训练中，该参数的设置无效。默认值：False。
    - **profile_memory** (bool, 可选) – （仅限Ascend）表示是否收集Tensor内存数据。当值为True时，收集这些数据。默认值：False。
    - **start_profile** (bool, 可选) – 该参数控制是否在Profiler初始化的时候开启数据采集。默认值：True。

    **异常：**

    - **RuntimeError** – 当CANN的版本与MindSpore版本不匹配时，生成的ascend_job_id目录结构MindSpore无法解析。

    .. py:method:: analyse()

        收集和分析训练的性能数据，支持在训练中和训练后调用。样例如上所示。

    .. py:method:: op_analyse(op_name, device_id)

        获取primitive类型的算子性能数据。

        **参数：**

        - **op_name** (str 或 list) – 表示要查询的primitive算子类型。
        - **device_id** (int, 可选) – 设备卡号，表示指定解析哪张卡的算子性能数据。在网络训练或者推理时使用，该参数可选。基于离线数据解析使用该接口时，默认值：0。

        **异常：**

        - **TypeError** – op_name参数类型不正确。
        - **TypeError** – device_id参数类型不正确。
        - **RuntimeError** – 在Ascend上使用该接口获取性能数据。

        **支持平台：**

        ``GPU`` ``CPU``

    .. py:method:: start()

        开启Profiler数据采集，可以按条件开启Profiler。

        **异常：**

        - **RuntimeError** – profiler已经开启。
        - **RuntimeError** – 停止Minddata采集后，不支持重复开启。
        - **RuntimeError** – 如果start_profile参数未设置或设置为True。

    .. py:method:: stop()

        停止Profiler，可以按条件停止Profiler。

        **异常：**

        - **RuntimeError** – profiler没有开启。
