mindspore.Profiler
========================

.. py:class:: mindspore.Profiler(**kwargs)

    当前接口即将弃用，请使用 :class:`mindspore.profiler.profile` 代替。
    MindSpore用户能够通过该类对神经网络的性能进行采集。首先通过创建Profiler初始化对象开始分析，然后使用 `Profiler.analyse()` 停止收集并分析结果。可通过 `MindStudio Insight <https://www.hiascend.com/developer/download/community/result?module=pt+sto+cann>`_ 工具可视化分析结果。目前，Profiler支持AICORE算子、AICPU算子、HostCPU算子、内存、设备通信、集群等数据的分析。

    参数：
        - **start_profile** (bool, 可选) - 该参数控制是否在Profiler初始化时开启数据采集。默认值： ``True`` 。
        - **output_path** (str, 可选) - 表示输出数据的路径。默认值： ``"./data"`` 。
        - **profiler_level** (ProfilerLevel, 可选) -（仅限Ascend）表示采集性能数据级别。默认值：``ProfilerLevel.Level0`` 。

          - ProfilerLevel.LevelNone：该设置仅在开启mstx时生效，表示不采集device侧的任何算子数据。
          - ProfilerLevel.Level0：最精简的采集性能数据级别，采集计算类算子的耗时数据和通信类大算子的基础数据。
          - ProfilerLevel.Level1：在Level0的基础上额外采集CANN层中AscendCL数据、AICORE性能数据以及通信类小算子数据。
          - ProfilerLevel.Level2：在Level1的基础上额外采集CANN层中GE和Runtime数据。

        - **activities** (list, 可选) - 表示需要收集的性能数据类型。默认值： ``[ProfilerActivity.CPU, ProfilerActivity.NPU]`` 。

          - ProfilerActivity.CPU：收集MindSpore框架数据。
          - ProfilerActivity.NPU：收集CANN软件栈和NPU数据。
          - ProfilerActivity.GPU：收集GPU数据。

        - **schedule** (schedule, 可选) - 设置采集的动作策略，由schedule类定义，需要配合step接口使用，默认值： ``None`` ，表示采集全部step的性能数据，详细介绍请参考 :class:`mindspore.profiler.schedule` 。
        - **on_trace_ready** (Callable, 可选) - 设置当性能数据采集完成时，执行的回调函数。默认值： ``None`` ，表示只采集，不解析性能数据，详细介绍请参考 :func:`mindspore.profiler.tensorboard_trace_handler` 。
        - **profile_memory** (bool, 可选) -（仅限Ascend）表示是否收集Tensor内存数据。当值为 ``True`` 时，收集这些数据。使用该参数时， `activities` 必须设置为 ``[ProfilerActivity.CPU, ProfilerActivity.NPU]`` 。在GE后端时收集算子内存数据，需要从第一个step开始采集。默认值： ``False`` ，该参数目前采集的算子名称不完整。将在后续版本修复，建议使用环境变量 ``MS_ALLOC_CONF`` 代替。
        - **aic_metrics** (AicoreMetrics, 可选) -（仅限Ascend）收集的AICORE性能数据类型，使用该参数时， `activities` 必须包含 ``ProfilerActivity.NPU`` ，且值必须包含在AicoreMetrics枚举值中，默认值： ``AicoreMetrics.AiCoreNone`` ，每种类型包含的数据项如下：

          - AicoreMetrics.AiCoreNone：不收集任何AICORE数据。
          - AicoreMetrics.ArithmeticUtilization：包含mac_fp16/int8_ratio、vec_fp32/fp16/int32_ratio、vec_misc_ratio等。
          - AicoreMetrics.PipeUtilization：包含vec_ratio、mac_ratio、scalar_ratio、mte1/mte2/mte3_ratio、icache_miss_rate等。
          - AicoreMetrics.Memory：包含ub\_read/write_bw、l1_read/write_bw、l2_read/write_bw、main_mem_read/write_bw等。
          - AicoreMetrics.MemoryL0：包含l0a_read/write_bw、l0b_read/write_bw、l0c_read/write_bw等。
          - AicoreMetrics.ResourceConflictRatio：包含vec_bankgroup/bank/resc_cflt_ratio等。
          - AicoreMetrics.MemoryUB：包含ub\_read/write_bw_mte、 ub\_read/write_bw_vector、 ub\_/write_bw_scalar等。
          - AicoreMetrics.L2Cache：包含write_cache_hit、 write_cache_miss_allocate、 r0_read_cache_hit、 r1_read_cache_hit等。本功能仅支持Atlas A2 训练系列产品。
          - AicoreMetrics.MemoryAccess：主存以及L2 Cache的存访带宽和存量统计。

        - **with_stack** (bool, 可选) - （仅限Ascend）表示是否收集Python侧的调用栈的数据，此数据在timeline中采用火焰图的形式呈现，使用该参数时， `activities` 必须包含 ``ProfilerActivity.CPU`` 。默认值： ``False`` 。
        - **data_simplification** (bool, 可选) - （仅限Ascend）是否开启数据精简，开启后仅保留profiler的交付件以及PROF_XXX目录下的原始性能数据，以节省空间。默认值: ``True`` 。
        - **l2_cache** (bool, 可选) - （仅限Ascend）是否收集L2 Cache数据，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。该采集项在ASCEND_PROFILER_OUTPUT文件夹下生成l2_cache.csv文件。在GE后端下，仅支持schedule配置中wait和skip_first参数都为0的场景。
        - **hbm_ddr** (bool, 可选) -（仅限Ascend）是否收集片上内存/DDR内存读写速率数据，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。
        - **pcie** (bool, 可选) -（仅限Ascend）是否收集PCIe带宽数据，当值为 ``True`` 时，收集这些数据。默认值： ``False`` 。
        - **data_process** (bool, 可选) -（Ascend/GPU）表示是否收集数据准备性能数据，默认值： ``False`` 。
        - **parallel_strategy** (bool, 可选) -（仅限Ascend）表示是否收集并行策略性能数据，默认值： ``False`` 。
        - **sync_enable** (bool, 可选) -（仅限GPU）Profiler是否用同步的方式收集算子耗时，默认值： ``True`` 。

          - True：同步方式，在把算子发送到GPU之前，在CPU端记录开始时间戳。然后在算子执行完毕返回到CPU端后，再记录结束时间戳。算子耗时为两个时间戳的差值。
          - False：异步方式，算子耗时为从CPU发送到GPU的耗时。这种方式能减少因增加Profiler对整体训练时间的影响。

    异常：
        - **RuntimeError** - 当CANN的版本与MindSpore版本不匹配时，MindSpore无法解析生成的ascend_job_id目录结构。

    .. py:method:: add_metadata(key, value)

        上报自定义metadata键值对数据。

        参数：
            - **key** (str) - metadata键值对的key。
            - **value** (str) - metadata键值对的value。

    .. py:method:: add_metadata_json(key, value)

        上报自定义metadata键值对value为json字符串数据。

        参数：
            - **key** (str) - metadata键值对的key。
            - **value** (str) - metadata键值对的value，格式为json字符串。

    .. py:method:: analyse(offline_path=None, pretty=False, step_list=None, mode="sync")

        收集和分析训练的性能数据，支持在训练中和训练后调用。样例如上所示。

        参数：
            - **offline_path** (Union[str, None], 可选) - 需要使用离线模式进行分析的数据路径。离线模式用于非正常退出场景。对于在线模式，该参数应设置为 ``None`` 。默认值： ``None`` 。
            - **pretty** (bool, 可选) - 对json文件进行格式化处理。该参数默认值为 ``False``，即不进行格式化。
            - **step_list** (list, 可选) - 只解析指定step的性能数据，指定的step必须是连续的整数。该参数默认值为 ``None``，即进行全解析。
            - **mode** (str, 可选) - 解析模式，同步解析或异步解析，可选参数为["sync", "async"]，默认值为 ``"sync"``。

              - "sync"：同步模式解析性能数据，会阻塞当前进程。
              - "async"：异步模式，另起一个子进程解析性能数据，不会阻塞当前进程。由于解析进程会额外占用CPU资源，请根据实际资源情况开启该模式。

    .. py:method:: offline_analyse(path, pretty=False, step_list=None, data_simplification=True)
        :classmethod:

        离线分析训练的性能数据，性能数据采集结束后调用。

        参数：
            - **path** (str) - 需要进行离线分析的profiling数据路径，指定到Profiler上层目录。
            - **pretty** (bool, 可选) - 对json文件进行格式化处理。该参数默认值为 ``False``，即不进行格式化。
            - **step_list** (list, 可选) - 只解析指定step的性能数据，指定的step必须是连续的整数。该参数默认值为 ``None``，即进行全解析。
            - **data_simplification** (bool, 可选) - 数据精简开关功能。默认值为 ``True``，即开启数据精简。

    .. py:method:: op_analyse(op_name, device_id=None)

        获取primitive类型的算子性能数据。

        参数：
            - **op_name** (str 或 list) - 表示要查询的primitive算子类型。
            - **device_id** (int, 可选) - 设备卡号，表示指定解析哪张卡的算子性能数据。在网络训练或者推理时使用，该参数可选。基于离线数据解析使用该接口时，默认值： ``None`` 。

        异常：
            - **TypeError** - `op_name` 参数类型不正确。
            - **TypeError** - `device_id` 参数类型不正确。
            - **RuntimeError** - 在Ascend上使用该接口获取性能数据。

    .. py:method:: start()

        开启Profiler数据采集。可以按条件开启Profiler。

        异常：
            - **RuntimeError** - Profiler已经开启。
            - **RuntimeError** - 如果 `start_profile` 参数未设置或设置为 ``True`` 。

    .. py:method:: step()

        用于在Ascend设备上，通过schedule和on_trace_ready区分步骤收集和解析性能数据。

        异常：
            - **RuntimeError** - 如果 `start_profile` 参数未设置或Profiler未开启。
            - **RuntimeError** - 如果 `schedule` 参数未设置。

    .. py:method:: stop()

        停止Profiler。可以按条件停止Profiler。

        异常：
            - **RuntimeError** - Profiler没有开启。
