mindspore.set_context
======================

.. py:function:: mindspore.set_context(**kwargs)

    设置运行环境的context，此接口将在后续版本中废弃，参数相关功能将通过新API接口提供。

    参数：
        - **mode** (int) - GRAPH_MODE（用0表示）或PYNATIVE_MODE（用1表示）。默认 ``PYNATIVE_MODE`` 。
        - **device_id** (int) - 目标设备的ID，默认 ``0`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.set_device` 代替。
        - **device_target** (str) - 程序运行的目标设备，支持 ``"Ascend"``、 ``"GPU"`` 和 ``"CPU"``。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.set_device` 代替。
        - **deterministic** (str) - 算子确定性计算，默认 ``"OFF"`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.set_deterministic` 代替。
        - **max_call_depth** (int) - 函数调用最大深度，默认 ``1000`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.set_recursion_limit` 代替。
        - **variable_memory_max_size** (str) - 此参数将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.set_memory` 代替。
        - **mempool_block_size** (str) - 设置设备内存池的块大小，默认 ``"1GB"`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.set_memory` 代替。
        - **memory_optimize_level** (str) - 内存优化级别，默认 ``"O0"`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.set_memory` 代替。
        - **max_device_memory** (str) - 设置设备可用的最大内存，默认 ``"1024GB"`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.set_memory` 代替。
        - **pynative_synchronize** (bool) - 是否启动设备同步执行，默认 ``False`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.runtime.launch_blocking` 代替。
        - **compile_cache_path** (str) - 保存编译缓存的路径，默认 ``"."`` 。此参数将在后续版本中废弃，请使用环境变量 `MS_COMPILER_CACHE_PATH` 代替。
        - **inter_op_parallel_num** (int) - 算子间并行数控制，默认 ``0`` 。此参数将在后续版本中废弃。请使用接口 :func:`mindspore.runtime.dispatch_threads_num` 代替。
        - **disable_format_transform** (bool) - 是否取消NCHW到NHWC的自动格式转换功能，默认 ``False`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.jit` 相关参数代替。
        - **jit_syntax_level** (int) - 设置jit语法支持级别，默认 ``LAX`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.jit` 的 `fullgraph` 参数代替。 `fullgraph` 参数被设置为 ``True`` 等同于 `jit_syntax_level` 参数被设置为 ``STRICT`` ， `fullgraph` 参数被设置为 ``False`` 等同于 `jit_syntax_level` 参数被设置为 ``LAX``。
        - **jit_config** (dict) - 设置全局编译选项。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.jit` 相关参数代替。
        - **exec_order** (str) - 算子执行时的排序方法，此参数将在后续版本中废弃，请使用接口 :func:`mindspore.jit` 相关参数代替。
        - **op_timeout** (int) - 设置一个算子的最大执行时间。默认 ``900`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.ascend.op_debug.execute_timeout` 代替。
        - **aoe_tune_mode** (str) - AOE调优。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.ascend.op_tuning.aoe_tune_mode` 代替。
        - **aoe_config** (dict) - aoe专用参数。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.ascend.op_tuning.aoe_job_type` 代替。
        - **runtime_num_threads** (int) - 运行时actor和CPU算子核使用的线程池线程数，默认 ``30`` 。此参数将在后续版本中废弃。请使用接口 :func:`mindspore.device_context.cpu.op_tuning.threads_num` 代替。
        - **save_graphs** (bool 或 int) - 表示是否保存中间编译图。默认 ``0`` 。此参数将在后续版本中废弃，请使用环境变量 `MS_DEV_SAVE_GRAPHS` 代替。
        - **save_graphs_path** (str) - 表示保存计算图的路径。默认 ``"."`` 。此参数将在后续版本中废弃，请使用环境变量 `MS_DEV_SAVE_GRAPHS_PATH` 代替。
        - **precompile_only** (bool) - 是否仅预编译网络，默认 ``False`` 。此参数将在后续版本中废弃，请使用环境变量 `MS_DEV_PRECOMPILE_ONLY` 代替。
        - **enable_compile_cache** (bool) - 是否加载或者保存图编译缓存，默认 ``False`` 。此参数将在后续版本中废弃，请使用环境变量 `MS_COMPILER_CACHE_ENABLE` 代替。
        - **ascend_config** (dict) - 设置Ascend硬件平台参数。

          - **precision_mode** (str): 混合精度模式设置。此参数将在后续版本中废弃，默认 ``"force_fp16"`` 。请使用接口 :func:`mindspore.device_context.ascend.op_precision.precision_mode` 代替。
          - **jit_compile** (bool): 是否在线编译。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.ascend.op_tuning.op_compile` 代替。
          - **matmul_allow_hf32** (bool): 是否为Matmul类算子使能FP32转换为HF32，默认 ``False`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.ascend.op_precision.matmul_allow_hf32` 代替。
          - **conv_allow_hf32** (bool): 是否为Conv类算子使能FP32转换为HF32，默认 ``True`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.ascend.op_precision.conv_allow_hf32` 代替。
          - **op_precision_mode** (str): 算子精度模式配置文件路径。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.ascend.op_precision.op_precision_mode` 代替。
          - **op_debug_option** (str): Ascend算子调试配置。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.ascend.op_debug.debug_option` 代替。
          - **ge_options** (dict): 设置CANN的options配置项。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.jit` 代替。
          - **atomic_clean_policy** (int): 清理网络中atomic算子占用的内存的策略。默认 ``1`` , 不集中清理。 ``0`` 表示集中清理。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.jit` 相关参数代替。
          - **exception_dump** (str): 开启Ascend算子异常dump。默认 ``"2"``。此参数已废弃，请使用接口 :func:`mindspore.device_context.ascend.op_debug.aclinit_config` 代替。
          - **host_scheduling_max_threshold** (int): 控制根图是否使用动态shape调度的最大阈值，默认 ``0`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.jit` 相关参数代替。
          - **parallel_speed_up_json_path** (Union[str, None]): 并行加速配置文件，此参数将在后续版本中废弃，请使用接口 :func:`mindspore.parallel.auto_parallel.AutoParallel.transformer_opt` 代替。
          - **hccl_watchdog** (bool): 开启一个线程监控集合通信故障。默认 ``True`` 。此参数将在后续版本中废弃，请使用环境变量 `MS_ENABLE_THM="{HCCL_WATCHDOG:1}"` 控制该功能开启。
        - **gpu_config** (dict) - 设置GPU硬件平台专用参数，默认不设置。

          - **conv_fprop_algo** (str): 指定cuDNN的卷积前向算法，默认 ``"normal"`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.gpu.op_tuning.conv_fprop_algo` 代替。
          - **conv_dgrad_algo** (str): 指定cuDNN的卷积输入数据的反向算法，默认 ``"normal"`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.gpu.op_tuning.conv_dgrad_algo` 代替。
          - **conv_wgrad_algo** (str): 指定cuDNN的卷积输入卷积核的反向算法，默认 ``"normal"`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.gpu.op_tuning.conv_wgrad_algo` 代替。
          - **conv_allow_tf32** (bool): 是否开启卷积在cuDNN下的TF32张量核计算，默认 ``True`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.gpu.op_precision.conv_allow_tf32` 代替。
          - **matmul_allow_tf32** (bool): 是否开启矩阵乘在CUBLAS下的TF32张量核计算，默认 ``False`` 。此参数将在后续版本中废弃，请使用接口 :func:`mindspore.device_context.gpu.op_precision.matmul_allow_tf32` 代替。
        - **print_file_path** (str) - 此参数将在后续版本中废弃。
        - **env_config_path** (str) - 此参数将在后续版本中废弃。
        - **debug_level** (int) - 此参数将在后续版本中废弃。
        - **reserve_class_name_in_scope** (bool) - 此参数将在后续版本中废弃。
        - **check_bprop** (bool) - 此参数将在后续版本中废弃。
        - **enable_reduce_precision** (bool) - 此参数将在后续版本中废弃。
        - **grad_for_scalar** (bool) - 此参数将在后续版本中废弃。
        - **support_binary** (bool) - 是否支持在图模式下运行.pyc或.so，默认 ``False`` 。此参数将在后续版本中废弃，请使用环境变量 `MS_SUPPORT_BINARY` 代替。
