mindspore.set_context
======================

.. py:function:: mindspore.set_context(**kwargs)

    设置运行环境的context。

    在运行程序之前，应配置context。如果没有配置，默认情况下将根据设备目标进行自动设置。

    .. note::
        设置属性时，必须输入属性名称。net初始化后不建议更改模式，因为一些操作的实现在Graph模式和PyNative模式下是不同的。默认值：PYNATIVE_MODE。

    某些配置适用于特定的设备，有关详细信息，请参见下表：

    +-------------------------+------------------------------+----------------------------+
    | 功能分类                |    配置参数                  |          硬件平台支持      |
    +=========================+==============================+============================+
    | 系统配置                |   device_id                  |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |   device_target              |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  max_device_memory           |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  variable_memory_max_size    |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  mempool_block_size          |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  op_timeout                  |  Ascend                    |
    +-------------------------+------------------------------+----------------------------+
    | 调试配置                |  save_graphs                 |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  save_graphs_path            |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  enable_dump                 |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  save_dump_path              |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  deterministic               |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  print_file_path             |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  env_config_path             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  precompile_only             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  reserve_class_name_in_scope |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  pynative_synchronize        |  GPU/Ascend                |
    +-------------------------+------------------------------+----------------------------+
    | 执行控制                |   mode                       |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  enable_graph_kernel         |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  graph_kernel_flags          |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  enable_reduce_precision     |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  auto_tune_mode              |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  check_bprop                 |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  max_call_depth              |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  grad_for_scalar             |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  enable_compile_cache        |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  inter_op_parallel_num       |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  runtime_num_threads         |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  compile_cache_path          |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  disable_format_transform    |  GPU                       |
    |                         +------------------------------+----------------------------+
    |                         |  support_binary              |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  memory_optimize_level       |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  memory_offload              |  GPU/Ascend                |
    |                         +------------------------------+----------------------------+
    |                         |  ascend_config               |  Ascend                    |
    +-------------------------+------------------------------+----------------------------+

    参数：
        - **device_id** (int) - 表示目标设备的ID，其值必须在[0, device_num_per_host-1]范围中，且 `device_num_per_host` 的值不应超过4096。默认值：0。
        - **device_target** (str) - 表示待运行的目标设备，支持 'Ascend'、 'GPU'和 'CPU'。如果未设置此参数，则使用MindSpore包对应的后端设备。
        - **max_device_memory** (str) - 设置设备可用的最大内存。格式为"xxGB"。默认值：1024GB。实际使用的内存大小是设备的可用内存和 `max_device_memory` 值中的最小值。
        - **variable_memory_max_size** (str) - 此参数已弃用，将被删除。请使用 `max_device_memory` 。
        - **mempool_block_size** (str) - 设置设备内存池的块大小。格式为"xxGB"。默认值：1GB。最小值是1GB。实际使用的内存池块大小是设备的可用内存和 `mempool_block_size` 值中的最小值。
        - **op_timeout** (int) - 设置一个算子的最大执行时间，以秒为单位。如果执行时间超过这个值，系统将终止该任务。0意味着无限等待。默认值：600。
        - **save_graphs** (bool 或 int) - 表示是否保存中间编译图。默认值：0。可用的选项为：

          - False或0：不保存中间编译图。
          - 1：运行时会输出图编译过程中生成的一些中间文件。
          - True或2：生成更多后端流程相关的ir文件。
          - 3：生成可视化计算图和更多详细的前端ir图。
          
          当 `save_graphs` 属性设为1、2、3或者True时， `save_graphs_path` 属性用于设置中间编译图的存储路径。默认情况下，计算图保存在当前目录下。
        - **save_graphs_path** (str) - 表示保存计算图的路径。默认值："."。如果指定的目录不存在，系统将自动创建该目录。在分布式训练中，图形将被保存到 `save_graphs_path/rank_${rank_id}/` 目录下。 `rank_id` 为集群中当前设备的ID。
        - **enable_dump** (bool) - 此参数已弃用，将在下一版本中删除。
        - **save_dump_path** (str) - 此参数已弃用，将在下一版本中删除。
        - **deterministic** (str) - 表示是否使能算子确定性运行模式。值必须在['ON','OFF']范围内，默认值：'OFF'。

          - ON：开启算子确定性运行模式。
          - OFF：关闭算子确定性运行模式。

          当确定性开启时，模型中的算子将在Ascend中具有确定性。这意味着，如果算子在同一硬件上使用相同的输入运行多次，则每次都会有完全相同的输出。这对于调试模型很有用。
        - **print_file_path** (str) - 该路径用于保存打印数据。使用时 :class:`mindspore.ops.Print` 可以打印输入的张量或字符串信息，使用方法 :func:`mindspore.parse_print` 解析保存的文件。如果设置了此参数，打印数据保存到文件，未设置将显示到屏幕。如果保存的文件已经存在，则将添加时间戳后缀到文件中。将数据保存到文件解决了屏幕打印中的数据丢失问题，如果未设置，将报告错误:"prompt to set the upper absolute path"。
        - **env_config_path** (str) - 通过 `mindspore.set_context(env_config_path="./mindspore_config.json")` 来设置MindSpore环境配置文件路径。

          配置Running Data Recorder：

          - **enable**：表示在发生故障时是否启用Running Data Recorder去收集和保存训练中的关键数据。设置为True时，将打开Running Data Recorder。设置为False时，将关闭Running Data Recorder。
          - **mode**：设置导出数据时的RDR模式。当设置为1时，RDR只在故障情况下输出数据。当设置为2时，RDR在故障情况和正常结束情况下输出数据。默认值：1。
          - **path**：设置Running Data Recorder保存数据的路径。当前路径必须是一个绝对路径。

          内存重用：

          - **mem_Reuse**：表示内存复用功能是否打开。设置为True时，将打开内存复用功能。设置为False时，将关闭内存复用功能。
            有关running data recoder和内存复用配置详细信息，请查看 `配置RDR和内存复用 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/custom_debug.html>`_。

        - **precompile_only** (bool) - 表示是否仅预编译网络。默认值：False。设置为True时，仅编译网络，而不执行网络。
        - **reserve_class_name_in_scope** (bool) - 表示是否将网络类名称保存到所属ScopeName中。默认值：True。每个节点都有一个ScopeName。子节点的ScopeName是其父节点。如果 `reserve_class_name_in_scope` 设置为True，则类名将保存在ScopeName中的关键字"net-"之后。例如：

          Default/net-Net1/net-Net2 (reserve_class_name_in_scope=True)

          Default/net/net (reserve_class_name_in_scope=False)

        - **pynative_synchronize** (bool) - 表示是否在PyNative模式下启动设备同步执行。默认值：False。设置为False时，将在设备上异步执行算子。当算子执行出错时，将无法定位特定错误脚本代码的位置。当设置为True时，将在设备上同步执行算子。这将降低程序的执行性能。此时，当算子执行出错时，可以根据错误的调用栈来定位错误脚本代码的位置。
        - **mode** (int) - 表示在GRAPH_MODE(0)或PYNATIVE_MODE(1)模式中运行，两种模式都支持所有后端。默认值：PYNATIVE_MODE。
        - **enable_graph_kernel** (bool) - 表示开启图算融合去优化网络执行性能。默认值：False。如果 `enable_graph_kernel` 设置为True，则可以启用加速。有关图算融合的详细信息，请查看 `使能图算融合 <https://www.mindspore.cn/docs/zh-CN/master/design/graph_fusion_engine.html>`_ 。
        - **graph_kernel_flags** (str) - 图算融合的优化选项，当与enable_graph_kernel冲突时，它的优先级更高。其仅适用于有经验的用户。例如，mindspore.set_context(graph_kernel_flags="--opt_level=2 --dump_as_text")。一些常用选项：

          - **opt_level**：设置优化级别。默认值：2。当opt_level的值大于0时，启动图算融合。可选值包括：

            - 0：关闭图算融合。
            - 1：启动算子的基本融合。
            - 2：包括级别1的所有优化，并打开更多的优化，如CSE优化算法、算术简化等。
            - 3：包括级别2的所有优化，并打开更多的优化，如SitchingFusion、ParallelFusion等。在某些场景下，该级别的优化激进且不稳定。使用此级别时要小心。

          - **dump_as_text**：将关键过程的详细信息生成文本文件保存到"graph_kernel_dump"目录里。默认值：False。

            有关更多选项，可以参考实现代码。

        - **enable_reduce_precision** (bool) - 表示是否开启降低精度计算。默认值：True。设置为True时，不支持用户指定的精度，且精度将自动更改。设置为False时，如果未指定用例的精度，则会报错并退出。
        - **auto_tune_mode** (str) - 表示算子构建时的自动调整模式，以获得最佳的切分性能。默认值：NO_TUNE。其值必须在['RL', 'GA', 'RL,GA']范围中。

          - RL：强化学习调优。
          - GA：遗传算法调优。
          - RL，GA：当RL和GA优化同时打开时，工具会根据网络模型中的不同算子类型自动选择RL或GA。RL和GA的顺序没有区别。（自动选择）。

          有关启用算子调优工具设置的更多信息，请查看 `使能算子调优工具 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/auto_tune.html>`_。

        - **check_bprop** (bool) - 表示是否检查反向传播节点，以确保反向传播节点输出的形状(shape)和数据类型与输入参数相同。默认值：False。
        - **max_call_depth** (int) - 指定函数调用的最大深度。其值必须为正整数。默认值：1000。当嵌套Cell太深或子图数量太多时，需要设置 `max_call_depth` 参数。系统最大堆栈深度应随着 `max_call_depth` 的调整而设置为更大的值，否则可能会因为系统堆栈溢出而引发 "core dumped" 异常。
        - **grad_for_scalar** (bool) - 表示是否获取标量梯度。默认值：False。当 `grad_for_scalar` 设置为True时，则可以导出函数的标量输入。由于后端目前不支持伸缩操作，所以该接口只支持在前端可推演的简单操作。
        - **enable_compile_cache** (bool) - 表示是否加载或者保存前端编译的图。当 `enable_compile_cache` 被设置为True时，在第一次执行的过程中，一个硬件无关的编译缓存会被生成并且导出为一个MINDIR文件。当该网络被再次执行时，如果 `enable_compile_cache` 仍然为True并且网络脚本没有被更改，那么这个编译缓存会被加载。注意目前只支持有限的Python脚本更改的自动检测，这意味着可能有正确性风险。默认值：False。这是一个实验特性，可能会被更改或者删除。
        - **compile_cache_path** (str) - 保存前端图编译缓存的路径。默认值："."。如果目录不存在，系统会自动创建这个目录。缓存会被保存到如下目录： `compile_cache_path/rank_${rank_id}/` 。 `rank_id` 是集群上当前设备的ID。
        - **inter_op_parallel_num** (int) - 算子间并行数控制。 默认值为0，表示由框架默认指定。
        - **runtime_num_threads** (int) - 运行时actor和CPU算子核使用的线程池线程数，必须大于等于0。默认值为30，如果同时运行多个进程，应将该值设置得小一些，以避免线程争用。
        - **disable_format_transform** (bool) - 表示是否取消NCHW到NHWC的自动格式转换功能。当fp16的网络性能不如fp32的时，可以设置 `disable_format_transform` 为True，以尝试提高训练性能。默认值：False。
        - **support_binary** (bool) - 是否支持在图形模式下运行.pyc或.so。如果要支持在图形模式下运行.so或.pyc，可将 `support_binary` 置为True，并运行一次.py文件，从而将接口源码保存到接口定义.py文件中，因此要保证该文件可写。然后将.py文件编译成.pyc或.so文件，即可在图模式下运行。
        - **memory_optimize_level** (str) - 内存优化级别，默认值：O0。其值必须在 ['O0', 'O1'] 范围中。

          - O0: 执行性能优先，关闭 SOMAS (Safe Optimized Memory Allocation Solver)。
          - O1: 内存性能优先，使能 SOMAS。
        - **memory_offload** (str) - 是否开启Offload功能，在内存不足场景下将空闲数据临时拷贝至Host侧内存。其值必须在['ON', 'OFF']范围中，默认值为 'OFF'。

          - ON：开启memory offload功能。在Ascend硬件平台，未设置环境变量“GRAPH_OP_RUN=1”时本参数不生效；设置memory_optimize_level='O1'时本参数不生效。
          - OFF：关闭memory offload功能。
        - **ascend_config** (dict) - 设置Ascend硬件平台专用的参数，默认不设置。当前只仅支持在Ascend910B硬件平台设置，其他平台不生效。 

          - **precision_mode** (str): 混合精度模式设置，Ascend910B硬件平台训练默认值：must_keep_origin_dtype，推理网络默认值：force_fp16。其值范围如下：

            - force_fp16: 当算子既支持float16，又支持float32，直接选择float16.
            - allow_fp32_to_fp16: 当算子不支持float32数据类型时，直接降低精度float16.
            - allow_mix_precision: 自动混合精度，针对全网算子，按照内置的优化策略，自动将部分算子的精度降低到float16或bfloat16.
            - must_keep_origin_dtype: 保持原图精度.
            - force_fp32: 当算子既支持float16，又支持float32，直接选择float32.
            - force_lowerprecision: 当算子支持float16或者bfloat16，又支持float32，直接选择float16或者bfloat16.
            - allow_fp32_to_bf16: 当算子不支持float32数据类型时，直接降低精度到bfloat16.
            - allow_fp32_to_lowprecision: 当算子不支持float32数据类型时，直接降低精度到float16或者bfloat16.
            - allow_mix_precision_fp16: 自动混合精度，正对全网算子，按照内置的优化策略，自动将部分算子的精度降低到float16.
            - allow_mix_precision_bf16: 自动混合精度，正对全网算子，按照内置的优化策略，自动将部分算子的精度降低到bfloat16.

          - **jit_compile** (bool): 表示是否选择在线编译。默认值：True。当设置为False时，优先选择系统中已经编译好的算子二进制文件，提升编译性能。

    异常：
        - **ValueError** - 输入key不是上下文中的属性。
