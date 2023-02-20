mindspore.Profiler
========================

.. py:class:: mindspore.Profiler(**kwargs)

    MindSpore用户能够通过该类对神经网络的性能进行采集。可以通过导入 `mindspore.Profiler` 然后初始化Profiler对象以开始分析，使用 `Profiler.analyse()` 停止收集和分析。可通过Mindinsight工具可视化分析结果。目前，Profiler支持AICORE算子、AICPU算子、HostCPU算子、内存、设备通信、集群等数据的分析。

    参数：
        - **output_path** (str, 可选) - 表示输出数据的路径。默认值："./data"。
        - **op_time** (bool, 可选) -（Ascend/GPU）表示是否收集算子性能数据，默认值：True。
        - **profile_communication** (bool, 可选) -（仅限Ascend）表示是否在多设备训练中收集通信性能数据。当值为True时，收集这些数据。在单台设备训练中，该参数的设置无效。使用此参数时，op_time必须设置成True。默认值：False。
        - **profile_memory** (bool, 可选) -（仅限Ascend）表示是否收集Tensor内存数据。当值为True时，收集这些数据。使用此参数时，op_time必须设置成True。默认值：False。
        - **parallel_strategy** (bool, 可选) -（仅限Ascend）表示是否收集并行策略性能数据， 默认值：True。
        - **start_profile** (bool, 可选) - 该参数控制是否在Profiler初始化的时候开启数据采集。默认值：True。
        - **aicore_metrics** (int, 可选) -（仅限Ascend）收集的AICORE性能数据类型。值必须包含在[-1, 0, 1, 2, 3, 4, 5]，默认值：0，每种类型包含的数据项如下：

          - -1: 不收集任何AICORE数据。
          - 0: ArithmeticUtilization，包含mac_fp16/int8_ratio、vec_fp32/fp16/int32_ratio、vec_misc_ratio等。
          - 1: PipeUtilization，包含vec_ratio、mac_ratio、scalar_ratio、mte1/mte2/mte3_ratio、icache_miss_rate等。
          - 2: Memory，包含ub\_read/write_bw、l1_read/write_bw、l2_read/write_bw、main_mem_read/write_bw等。
          - 3: MemoryL0，包含l0a_read/write_bw、l0b_read/write_bw、l0c_read/write_bw等。
          - 4: ResourceConflictRatio，包含vec_bankgroup/bank/resc_cflt_ratio等。
          - 5: MemoryUB，包含ub\_read/write_bw_mte, ub\_read/write_bw_vector, ub\_/write_bw_scalar等。

        - **l2_cache** (bool, 可选) -（仅限Ascend）是否收集l2缓存数据，当值为True时，收集这些数据。默认值：False。
        - **sync_enable** (bool, 可选) -（仅限GPU）Profiler是否用同步的方式收集算子耗时，默认值：True。

          - True: 同步方式，在把算子发送到GPU之前，在CPU端记录开始时间戳。然后在算子执行完毕返回到CPU端后，再记录结束时间戳。算子耗时为两个时间戳的差值。
          - False: 异步方式，算子耗时为从CPU发送到GPU的耗时。这种方式能减少因增加Profiler对训练时间的影响。

        - **data_process** (bool, 可选) -（Ascend/GPU）表示是否收集数据准备性能数据，默认值：True。
        - **timeline_limit** (int, 可选) - 设置限制timeline文件存储上限大小（单位M），使用此参数时，op_time必须设置成True。默认值：500。

    异常：
        - **RuntimeError** - 当CANN的版本与MindSpore版本不匹配时，生成的ascend_job_id目录结构MindSpore无法解析。

    .. py:method:: analyse()

        收集和分析训练的性能数据，支持在训练中和训练后调用。样例如上所示。

    .. py:method:: op_analyse(op_name, device_id=None)

        获取primitive类型的算子性能数据。

        参数：
            - **op_name** (str 或 list) - 表示要查询的primitive算子类型。
            - **device_id** (int, 可选) - 设备卡号，表示指定解析哪张卡的算子性能数据。在网络训练或者推理时使用，该参数可选。基于离线数据解析使用该接口时，默认值：0。

        异常：
            - **TypeError** - op_name参数类型不正确。
            - **TypeError** - device_id参数类型不正确。
            - **RuntimeError** - 在Ascend上使用该接口获取性能数据。

    .. py:method:: start()

        开启Profiler数据采集。可以按条件开启Profiler。

        异常：
            - **RuntimeError** - Profiler已经开启。
            - **RuntimeError** - 停止Minddata采集后，不支持重复开启。
            - **RuntimeError** - 如果start_profile参数未设置或设置为True。

    .. py:method:: stop()

        停止Profiler。可以按条件停止Profiler。

        异常：
            - **RuntimeError** - Profiler没有开启。
