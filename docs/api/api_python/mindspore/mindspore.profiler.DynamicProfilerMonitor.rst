mindspore.profiler.DynamicProfilerMonitor
=========================================

.. py:class:: mindspore.profiler.DynamicProfilerMonitor(cfg_path=None, output_path="./dyn_profile_data", poll_interval=2, **kwargs)

    该类用于动态采集MindSpore神经网络性能数据。

    参数：
        - **cfg_path** (str) - （仅限Ascend）动态Profiler的json配置文件的文件夹路径。要求该路径是能够被所有节点访问到的共享目录。json配置文件相关参数如下。

          - start_step (int, 必选) - 设置Profiler开始采集的步数，为相对值，训练的第一步为 ``1`` 。修改 `start_step` 时，需要满足新的 `start_step` 大于上一次采集时设置的 `stop_step` 。默认值： ``-1`` ，表示在整个训练流程不会开始采集。
          - stop_step (int, 必选) - 设置Profiler开始停止的步数，为相对值，训练的第一步为 ``1`` 。修改 `stop_step` 时，需要满足新的 `stop_step` 大于等于 `start_step` 。默认值： ``-1`` ，表示在整个训练流程不会开始采集。
          - aic_metrics (int/str, 可选) - 设置采集AI Core指标数据，当前版本可传入int或str任一类型，后续会更新为只传入str类型。其中 ``0`` 或 ``"PipeUtilization"`` 代表PipeUtilization； ``1`` 或 ``"ArithmeticUtilization"`` 代表ArithmeticUtilization； ``2`` 或 ``"Memory"`` 代表Memory； ``3`` 或 ``"MemoryL0"`` 代表MemoryL0； ``4`` 或 ``"MemoryUB"`` 代表MemoryUB； ``5`` 或 ``"ResourceConflictRatio"`` 代表ResourceConflictRatio； ``6`` 或 ``"L2Cache"`` 代表L2Cache； ``7`` 或 ``"MemoryAccess"`` 代表MemoryAccess。默认值： ``"AiCoreNone"`` ，表示不采集AI Core指标。
          - profiler_level (int/str, 可选) - 设置采集性能数据级别，当前版本可传入int或str任一类型，后续会更新为只传入str类型。其中 ``-1`` 或 ``"LevelNone"`` 代表ProfilerLevel.LevelNone， ``0`` 或 ``"Level0"`` 代表ProfilerLevel.Level0， ``1`` 或 ``"Level1"`` 代表ProfilerLevel.Level1， ``2`` 或 ``"Level2"`` 代表ProfilerLevel.Level2。默认值 ``"Level0"`` ，表示ProfilerLevel.Level0的采集级别。
          - activities (int/list, 可选) - 设置采集性能数据的设备，当前版本可传入int或list任一类型，后续会更新为只传入list类型。其中 ``0`` 或 ``["CPU","NPU"]`` 代表CPU+NPU， ``1`` 或 ``["CPU"]`` 代表CPU， ``2`` 或 ``["NPU"]`` 代表NPU。默认值 ``["CPU","NPU"]`` ，表示采集CPU+NPU的性能数据。
          - export_type (int/list, 可选) - 设置导出性能数据的类型，当前版本可传入int或list任一类型，后续会更新为只传入list类型。其中 ``0`` 或 ``["text"]`` 代表text， ``1`` 或 ``["db"]`` 代表db， ``2`` 或 ``["text","db"]`` 代表text和db。默认值 ``["text"]`` ，表示只导出text类型的性能数据。
          - profile_memory (bool, 可选) - 设置是否采集内存性能数据，true代表采集，false代表不采集。默认值false，表示不采集内存性能数据。
          - mstx (bool, 可选) - 设置是否开启mstx，true代表开启，false代表不开启。默认值false，表示不开启mstx。
          - analyse (bool, 可选) - 设置是否开启在线解析，true代表开启，false代表不开启。默认值false，表示不开启在线解析。该参数优先级高于analyse_mode参数，当该参数设置为false时，analyse_mode参数设置不生效。当该参数设置为true时，analyse_mode参数设置为-1不生效。
          - analyse_mode (int, 可选) - 设置在线解析的模式，0代表"sync"，1代表"async"。默认值-1，表示不使用在线解析。该参数优先级低于analyse参数，当analyse参数设置为false时，该参数设置不生效；当analyse参数设置为true时，该参数设置为-1不生效。
          - parallel_strategy (bool, 可选) - 设置是否采集并行策略性能数据，true代表采集，false代表不采集。默认值false，表示不采集并行策略性能数据。
          - with_stack (bool, 可选) - 设置是否采集调用栈信息，true代表采集，false代表不采集。默认值false，表示不采集调用栈。
          - data_simplification (bool, 可选) - 设置开启数据精简，true代表开启，false代表不开启。默认值true，表示开启数据精简。
          - mstx_domain_include (list, 可选) - mstx开关打开时设置使能的domain名称集合，且名称必须是str类型。默认值：``[]`` ，表示不使用该参数控制domain。该参数与mstx_domain_exclude参数互斥，不能同时设置。如果都设置，只有mstx_domain_include参数生效。
          - mstx_domain_exclude (list, 可选) - mstx开关打开时设置不使能的domain名称集合，且名称必须是str类型。默认值：``[]`` ，表示不使用该参数控制domain。
          - record_shapes (bool, 可选) - 设置是否采集算子输入tensor的shape信息，true代表采集，false代表不采集。默认值false，表示不采集算子输入tensor的shape信息。
          - prof_path (str, 可选) - 动态Profiler的输出文件路径。与接口参数 `output_path` 作用相同，两者同时配置时以 `prof_path` 为准。默认值：``"./"`` 。
          - sys_io (bool, 可选) - 设置是否采集NIC和RoCE数据。默认值： ``False`` ，表示不采集这些数据。
          - sys_interconnection (bool, 可选) - 设置是否采集系统互连数据，包括集合通信带宽数据（HCCS）、PCIe数据以及片间传输带宽信息。默认值： ``False`` ，表示不采集这些数据。
          - host_sys(list, 可选) - 采集Host侧的系统调用、存储类指标和CPU使用率数据，传入list类型，支持传入 ``"cpu"`` 、 ``"mem"`` 、 ``"disk"`` 、 ``"network"`` 、 ``"osrt"`` 中的一个或多个，其中 ``"cpu"`` 代表进程级别的cpu利用率， ``"mem"`` 代表进程级别的内存利用率， ``"disk"`` 代表进程级别的磁盘I/O利用率， ``"network"`` 代表系统级别的网络I/O利用率， ``"osrt"`` 代表系统级别的syscall和pthreadcall。默认值： ``[]`` ，表示不采集host侧系统类数据。在采集DISK或OSRT数据时，需要提前安装好iotop、perf、ltrace三方工具，详细步骤请参考 `安装三方工具 <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/atlasprofiling_16_0136.html>`_ ；安装三方工具成功后，需要配置用户权限，详细步骤请参考 `配置用户权限 <https://www.hiascend.com/document/detail/zh/mindstudio/80RC1/T&ITools/Profiling/atlasprofiling_16_0137.html>`_ ，注意在配置用户权限的第3步中，需要将msprof_data_collection.sh脚本中的内容替换为 `msprof_data_collection.sh <https://atomgit.com/mindspore/mindspore/blob/master/docs/api/api_python/mindspore/script/msprof_data_collection.sh>`_ 。

        - **output_path** (str, 可选) - （仅限Ascend）动态Profiler的输出文件路径。默认值：``"./dyn_profile_data"`` 。
        - **poll_interval** (int, 可选) - （仅限Ascend）监控进程的轮询周期，单位为秒。默认值：``2``。

    异常：
        - **RuntimeError** - 创建监控进程失败次数超过最大限制。

    .. py:method:: step()

        用于在Ascend设备上，区分step收集和解析性能数据。

        异常：
            - **RuntimeError** - 如果 `start_step` 参数设置大于 `stop_step` 参数设置。