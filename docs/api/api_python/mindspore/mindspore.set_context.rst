mindspore.set_context
======================

.. py:function:: mindspore.set_context(**kwargs)

    设置运行环境的context。

    在运行程序之前，应配置context。如果没有配置，默认情况下将根据设备目标进行自动设置。

    .. note::
        设置属性时，必须输入属性名称。net初始化后不建议更改模式，因为一些操作的实现在Graph模式和PyNative模式下是不同的。默认值： ``PYNATIVE_MODE`` 。

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
    |                         |  op_timeout                  |  GPU/Ascend                |
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
    |                         |  pynative_synchronize        |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  debug_level                 |  CPU/GPU/Ascend            |
    +-------------------------+------------------------------+----------------------------+
    | 执行控制                |   mode                       |   CPU/GPU/Ascend           |
    |                         +------------------------------+----------------------------+
    |                         |  enable_graph_kernel         |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  graph_kernel_flags          |  Ascend/GPU                |
    |                         +------------------------------+----------------------------+
    |                         |  enable_reduce_precision     |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  aoe_tune_mode               |  Ascend                    |
    |                         +------------------------------+----------------------------+
    |                         |  aoe_config                  |  Ascend                    |
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
    |                         +------------------------------+----------------------------+
    |                         |  jit_syntax_level            |  CPU/GPU/Ascend            |
    |                         +------------------------------+----------------------------+
    |                         |  gpu_config                  |  GPU                       |
    |                         +------------------------------+----------------------------+
    |                         |  jit_config                  |  CPU/GPU/Ascend            |
    +-------------------------+------------------------------+----------------------------+

    参数：
        - **device_id** (int) - 表示目标设备的ID，其值必须在[0, device_num_per_host-1]范围中，且 `device_num_per_host` 的值不应超过4096。默认值： ``0`` 。
        - **device_target** (str) - 表示待运行的目标设备，支持 'Ascend'、 'GPU'和 'CPU'。如果未设置此参数，则使用MindSpore包对应的后端设备。
        - **max_device_memory** (str) - 设置设备可用的最大内存。格式为"xxGB"。默认值： ``1024GB`` 。实际使用的内存大小是设备的可用内存和 `max_device_memory` 值中的最小值。 `max_device_memory` 需要在程序运行之前设置。
        - **variable_memory_max_size** (str) - 此参数已弃用，将被删除。请使用 `max_device_memory` 。
        - **mempool_block_size** (str) - 设置PyNative模式或者jit level为"O0"或"O1"时设备内存池的块大小。格式为"xxGB"。默认值： ``1GB`` 。最小值是1GB。实际使用的内存池块大小是设备的可用内存和 `mempool_block_size` 值中的最小值。
        - **op_timeout** (int) - 设置一个算子的最大执行时间，以秒为单位。如果执行时间超过这个值，系统将终止该任务。0意味着使用默认值，AI Core和AICPU算子在不同硬件上的默认值有差异，详细信息请查看 `昇腾社区关于aclrtSetOpExecuteTimeOut文档说明 <https://www.hiascend.com/en/document/detail/zh/CANNCommunityEdition/80RC1alpha003/apiref/appdevgapi/aclcppdevg_03_0228.html>`_。MindSpore默认设置值： ``900`` 。
        - **save_graphs** (bool 或 int) - 表示是否保存中间编译图。默认值： ``0`` 。可用的选项为：

          - False或0：不保存中间编译图。
          - 1：运行时会输出图编译过程中生成的一些中间文件。
          - True或2：生成更多后端流程相关的ir文件。
          - 3：生成可视化计算图和更多详细的前端ir图。

          当网络结构复杂时将 `save_graphs` 属性设为 ``2`` 或者 ``3`` 时可能会出现耗时过长的情况。如需要快速定位问题，可先设置 `save_graphs` 属性为 ``1`` 。

          当 `save_graphs` 属性设为 ``1`` 、 ``2`` 、 ``3`` 或者 ``True`` 时， `save_graphs_path` 属性用于设置中间编译图的存储路径。默认情况下，计算图保存在当前目录下。
        - **save_graphs_path** (str) - 表示保存计算图的路径。默认值： ``"."`` 。如果指定的目录不存在，系统将自动创建该目录。在分布式训练中，图形将被保存到 `save_graphs_path/rank_${rank_id}/` 目录下。 `rank_id` 为集群中当前设备的ID。
        - **deterministic** (str) - 表示是否使能算子确定性运行模式。值必须在['ON','OFF']范围内，默认值： ``'OFF'`` 。

          - ON：开启算子确定性运行模式。
          - OFF：关闭算子确定性运行模式。

          当确定性开启时，模型中的算子将在Ascend中具有确定性。这意味着，如果算子在同一硬件上使用相同的输入运行多次，则每次都会有完全相同的输出。这对于调试模型很有用。
        - **enable_dump** (bool) - 此参数已弃用，将在下一版本中删除。
        - **save_dump_path** (str) - 此参数已弃用，将在下一版本中删除。
        - **print_file_path** (str) - 该路径用于保存打印数据。使用时 :class:`mindspore.ops.Print` 可以打印输入的张量或字符串信息，使用方法 :func:`mindspore.parse_print` 解析保存的文件。如果设置了此参数，打印数据保存到文件，未设置将显示到屏幕。如果保存的文件已经存在，则将添加时间戳后缀到文件中。将数据保存到文件解决了屏幕打印中的数据丢失问题，如果未设置，将报告错误:"prompt to set the upper absolute path"。当print输出到文件时，单次print调用输出的总数据的大小不能超过2GB（受限于protobuf）。
        - **env_config_path** (str) - 通过 `mindspore.set_context(env_config_path="./mindspore_config.json")` 来设置MindSpore环境配置文件路径。

          配置Running Data Recorder：

          - **enable**：表示在发生故障时是否启用Running Data Recorder去收集和保存训练中的关键数据。设置为 ``True`` 时，将打开Running Data Recorder。设置为 ``False`` 时，将关闭Running Data Recorder。
          - **mode**：设置导出数据时的RDR模式。当设置为 ``1`` 时，RDR只在故障情况下输出数据。当设置为 ``2`` 时，RDR在故障情况和正常结束情况下输出数据。默认值： ``1`` 。
          - **path**：设置Running Data Recorder保存数据的路径。当前路径必须是一个绝对路径。

          内存重用：

          - **mem_Reuse**：表示内存复用功能是否打开。设置为 ``True`` 时，将打开内存复用功能。设置为 ``False`` 时，将关闭内存复用功能。

          配置详细信息，请查看 `Running Data Recorder <https://www.mindspore.cn/tutorials/experts/zh-CN/master/debug/rdr.html>`_ 和 `内存复用 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/optimize/mem_reuse.html>`_ 。

        - **precompile_only** (bool) - 表示是否仅预编译网络。默认值： ``False`` 。设置为 ``True`` 时，仅编译网络，而不执行网络。
        - **reserve_class_name_in_scope** (bool) - 表示是否将网络类名称保存到所属ScopeName中。默认值： ``True`` 。每个节点都有一个ScopeName。子节点的ScopeName是其父节点。如果 `reserve_class_name_in_scope` 设置为 ``True`` ，则类名将保存在ScopeName中的关键字"net-"之后。例如：

          Default/net-Net1/net-Net2 (reserve_class_name_in_scope=True)

          Default/net/net (reserve_class_name_in_scope=False)

        - **pynative_synchronize** (bool) - 表示是否在PyNative模式下启动设备同步执行。默认值： ``False`` 。设置为 ``False`` 时，将在设备上异步执行算子。当算子执行出错时，将无法定位特定错误脚本代码的位置。当设置为 ``True`` 时，将在设备上同步执行算子。这将降低程序的执行性能。此时，当算子执行出错时，可以根据错误的调用栈来定位错误脚本代码的位置。
        - **mode** (int) - 表示在GRAPH_MODE(0)或PYNATIVE_MODE(1)模式中运行，两种模式都支持所有后端。默认值： ``PYNATIVE_MODE`` 。
        - **enable_graph_kernel** (bool) - 表示开启图算融合去优化网络执行性能。默认值： ``False`` 。如果 `enable_graph_kernel` 设置为 ``True`` ，则可以启用加速。有关图算融合的详细信息，请查看 `使能图算融合 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/optimize/graph_fusion_engine.html>`_ 。
        - **graph_kernel_flags** (str) - 图算融合的优化选项，当与enable_graph_kernel冲突时，它的优先级更高。其仅适用于有经验的用户。例如：

          .. code-block::

              mindspore.set_context(graph_kernel_flags="--opt_level=2 --dump_as_text")

          一些常用选项：

          - **opt_level**：设置优化级别。默认值： ``2`` 。当opt_level的值大于0时，启动图算融合。可选值包括：

            - 0：关闭图算融合。
            - 1：启动算子的基本融合。
            - 2：包括级别1的所有优化，并打开更多的优化，如CSE优化算法、算术简化等。
            - 3：包括级别2的所有优化，并打开更多的优化，如SitchingFusion、ParallelFusion等。在某些场景下，该级别的优化激进且不稳定。使用此级别时要小心。

          - **dump_as_text**：将关键过程的详细信息生成文本文件保存到"graph_kernel_dump"目录里。默认值： ``False`` 。

        - **enable_reduce_precision** (bool) - 表示是否开启降低精度计算。默认值： ``True`` 。设置为 ``True`` 时，不支持用户指定的精度，且精度将自动更改。设置为 ``False`` 时，如果未指定用例的精度，则会报错并退出。
        - **aoe_tune_mode** (str) - 表示启动AOE调优，默认不设置。设置为 ``online`` 时，将启动在线调优，设置为 ``offline`` 时，将为离线调优保存GE图 。
        - **aoe_config** (dict) - 设置aoe工具专用的参数，默认不设置。

          - **job_type** (str): 设置调优类型，有算子调优和子图调优。默认为算子调优。

            - ``"1"``: 设置为子图调优。
            - ``"2"``: 设置为算子调优。

        - **check_bprop** (bool) - 表示是否检查反向传播节点，以确保反向传播节点输出的shape和数据类型与输入参数相同。默认值： ``False`` 。
        - **max_call_depth** (int) - 指定函数调用的最大深度。其值必须为正整数。默认值： ``1000`` 。当嵌套Cell太深或子图数量太多时，需要设置 `max_call_depth` 参数。系统最大堆栈深度应随着 `max_call_depth` 的调整而设置为更大的值，否则可能会因为系统堆栈溢出而引发 "core dumped" 异常。
        - **grad_for_scalar** (bool) - 表示是否获取标量梯度。默认值： ``False`` 。当 `grad_for_scalar` 设置为True时，则可以导出函数的标量输入。由于后端目前不支持伸缩操作，所以该接口只支持在前端可推演的简单操作。
        - **enable_compile_cache** (bool) - 表示是否加载或者保存前端编译的图。当 `enable_compile_cache` 被设置为True时，在第一次执行的过程中，一个硬件无关的编译缓存会被生成并且导出为一个MINDIR文件。当该网络被再次执行时，如果 `enable_compile_cache` 仍然为True并且网络脚本没有被更改，那么这个编译缓存会被加载。注意目前只支持有限的Python脚本更改的自动检测，这意味着可能有正确性风险。默认值： ``False`` 。这是一个实验特性，可能会被更改或者删除。
        - **compile_cache_path** (str) - 保存编译缓存的路径。默认值： ``"."`` 。如果目录不存在，系统会自动创建这个目录。缓存会被保存到如下目录： `compile_cache_path/rank_${rank_id}/` 。 `rank_id` 是集群上当前设备的ID。
        - **inter_op_parallel_num** (int) - 算子间并行数控制。 默认值为 ``0`` ，表示由框架默认指定。
        - **runtime_num_threads** (int) - 运行时actor和CPU算子核使用的线程池线程数，必须大于等于 ``0`` 。默认值为 ``30`` ，如果同时运行多个进程，应将该值设置得小一些，以避免线程争用。
        - **disable_format_transform** (bool) - 表示是否取消NCHW到NHWC的自动格式转换功能。当fp16的网络性能不如fp32的时，可以设置 `disable_format_transform` 为 ``True`` ，以尝试提高训练性能。默认值： ``False`` 。
        - **support_binary** (bool) - 是否支持在图形模式下运行.pyc或.so。如果要支持在图形模式下运行.so或.pyc，可将 `support_binary` 置为 ``True`` ，并运行一次.py文件，从而将接口源码保存到接口定义.py文件中，因此要保证该文件可写。然后将.py文件编译成.pyc或.so文件，即可在图模式下运行。
        - **memory_optimize_level** (str) - 内存优化级别，Ascend平台下默认值 ``O1``，其他平台默认值： ``O0`` 。其值必须在 ['O0', 'O1'] 范围中。

          - O0: 执行性能优先，关闭 SOMAS (Safe Optimized Memory Allocation Solver) 和一些其他内存优化。
          - O1: 内存性能优先，使能 SOMAS 和一些其他内存优化。
        - **memory_offload** (str) - 是否开启Offload功能，在内存不足场景下将空闲数据临时拷贝至Host侧内存。其值必须在['ON', 'OFF']范围中，默认值为 ``'OFF'`` 。

          - ON：开启memory offload功能。在Ascend硬件平台，在图编译等级不为O0时本参数不生效；设置memory_optimize_level='O1'时本参数不生效。
          - OFF：关闭memory offload功能。
        - **ascend_config** (dict) - 设置Ascend硬件平台专用的参数，默认不设置。
          precision_mode、jit_compile和atomic_clean_policy参数的默认值属于实验性质参数，将来可能会发生变化。

          - **precision_mode** (str): 混合精度模式设置。推理网络默认值： ``force_fp16`` 。其值范围如下：

            - force_fp16: 当算子既支持float16，又支持float32时，直接选择float16。
            - allow_fp32_to_fp16: 对于矩阵类算子，使用float16。对于矢量类算子，优先保持原图精度，如果网络模型中算子支持float32，则保留原始精度float32，如果网络模型中算子不支持float32，则直接降低精度到float16。
            - allow_mix_precision: 自动混合精度，针对全网算子，按照内置的优化策略，自动将部分算子的精度降低到float16或bfloat16。
            - must_keep_origin_dtype: 保持原图精度。
            - force_fp32: 当矩阵计算的算子输入为float16，输出既支持float16又支持float32时，强制转换成float32输出。
            - allow_fp32_to_bf16: 对于矩阵类算子，使用bfloat16。对于矢量类算子，优先保持原图精度，如果网络模型中算子支持float32，则保留原始精度float32，如果网络模型中算子不支持float32，则直接降低精度到bfloat16。
            - allow_mix_precision_fp16: 自动混合精度，针对全网算子，按照内置的优化策略，自动将部分算子的精度降低到float16。
            - allow_mix_precision_bf16: 自动混合精度，针对全网算子，按照内置的优化策略，自动将部分算子的精度降低到bfloat16。

          - **jit_compile** (bool): 表示是否选择在线编译。当设置为 ``True`` 时，优先选择在线编译，当设置为 ``False`` 时，优先选择系统中已经编译好的算子二进制文件，提升编译性能。默认设置为静态shape选择在线编译，动态shape选择算子二进制文件。
          - **atomic_clean_policy** (int): 表示清理网络中atomic算子占用的内存的策略。默认值： ``1`` 。

            - 0：集中清理网络中所有atomic算子占用的内存。
            - 1：不集中清理内存，对网络中每一个atomic算子进行单独清零。当网络中内存超限时，可以尝试此种清理方式，但可能会导致一定的性能损耗。

          - **matmul_allow_hf32** (bool): 是否为Matmul类算子使能FP32转换为HF32。默认值： ``False``。这是一个实验特性，可能会被更改或者删除。如果您想了解更多详细信息，
            请查询 `昇腾社区 <https://www.hiascend.com/>`_ 了解。
          - **conv_allow_hf32** (bool): 是否为Conv类算子使能FP32转换为HF32。默认值： ``True``。这是一个实验特性，可能会被更改或者删除。如果您想了解更多详细信息，
            请查询 `昇腾社区 <https://www.hiascend.com/>`_ 了解。
          - **exception_dump** (str): 开启Ascend算子异常dump，提供计算异常时候的输入输出信息。可以为 ``"0"``，``"1"``，``"2"``。为 ``"0"`` 时关闭异常dump；为 ``"1"`` 时dump出AICore异常算子输入输出数据；为 ``"2"`` 时dump出AICore异常算子输入数据，保存信息减少，但可提升性能。默认值： ``"2"``。
          - **op_precision_mode** (str): 算子精度模式配置文件的所在路径。如果您想了解更多详细信息, 请查询 `昇腾社区 <https://www.hiascend.com/>`_ 了解。
          - **op_debug_option** (str): 表示Ascend算子调试配置，默认不开启，当前只支持内存访问越界检测，可配置为 ``oom`` 。

            - ``oom`` : 涉及从全局内存中读写数据，例如读写算子数据等，该选项开启全局内存访问越界检测，实际执行算子时，若出现内存越界，AscendCL会返回 ``EZ9999`` 错误码。

          - **ge_options** (dict): 设置CANN的options配置项，配置项分为 ``global`` 和 ``session`` 二类 。这是一个实验特性，可能会被更改或者删除。
            详细的配置请查询 `options配置说明 <https://www.hiascend.com/document/detail/zh/canncommercial/70RC1/inferapplicationdev/graphdevg/atlasgeapi_07_0119.html>`_ 。
            `ge_options` 中的配置项可能与 `ascend_config` 中的配置项重复，若同时设置了 `ascend_config` 和 `ge_options` 中的相同配置项，则以 `ge_options` 中设置的为准。

            - global (dict): 设置global类的选项。
            - session (dict): 设置session类的选项。

          - **parallel_speed_up_json_path** (Union[str, None]): 并行加速配置文件，配置项可以参考 `parallel_speed_up.json <https://gitee.com/mindspore/mindspore/blob/master/config/parallel_speed_up.json>`_ 。
            当设置为None时，表示不启用。

            - **recompute_comm_overlap** (bool): 为 ``True`` 时表示开启反向重计算和通信掩盖。默认值： ``False`` 。
            - **matmul_grad_comm_overlap** (bool): 为 ``True`` 时表示开启反向Matmul和通信掩盖。默认值： ``False`` 。
            - **enable_task_opt** (bool): 为 ``True`` 时表示开启通信融合进行通信算子task数量优化。默认值： ``False`` 。
            - **enable_grad_comm_opt** (bool): 为 ``True`` 时表示开启梯度dx计算与数据并行梯度通信的掩盖，暂时不支持 `LazyInline <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.lazy_inline.html>`_ 功能下开启。默认值： ``False`` 。
            - **enable_opt_shard_comm_opt** (bool): 为 ``True`` 时表示开启正向计算与优化器并行的AllGather通信的掩盖，暂时不支持 `LazyInline <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore/mindspore.lazy_inline.html>`_ 功能下开启。默认值： ``False`` 。
            - **enable_concat_eliminate_opt** (bool): 为 ``True`` 时表示开启Concat消除优化，当前在开启细粒度双副本优化时有收益。默认值： ``False`` 。
            - **enable_begin_end_inline_opt** (bool): 为 ``True`` 时表示开启首尾micro_batch子图的内联，用于半自动并行子图模式，流水线并行场景，一般需要和其它通信计算掩盖优化一起使用。默认值： ``False`` 。
            - **compute_communicate_fusion_level** (int): 控制通算融合的级别。默认值：``0``。注：此功能需要配套Ascend Training Solution 24.0.RC2以上版本使用。

              - 0: 不启用通算融合。

              - 1: 仅对前向节点使能通算融合。

              - 2: 仅对反向节点使能通算融合。

              - 3: 对所有节点使能通算融合。
            - **bias_add_comm_swap** (bool): 为 ``True`` 时表示开启matmul-add结构下，通信算子与add算子执行顺序互换。当前仅支持bias为一维的情况。默认值： ``False`` 。
          - **host_scheduling_max_threshold** (int): 控制静态小图（根图）执行时是否使用动态shape调度的最大阈值，默认阈值为0。如果静态根图节点个数小于最大阈值，则使用动态shape调度。大模型场景，该方式可以节约stream资源。如果静态根图节点个数大于最大阈值，则保持原有流程不变。

        - **jit_syntax_level** (int) - 当通过GRAPH_MODE或者@jit装饰器触发图编译时，此选项用于设置JIT语法支持级别。
          其值必须为 ``STRICT`` 或 ``LAX`` ，默认值为 ``LAX`` 。全部级别都支持所有后端。

          - ``STRICT`` : 仅支持基础语法，且执行性能最佳。可用于MindIR导入导出。
          - ``LAX`` : 最大程度地兼容Python所有语法。执行性能可能会受影响，不是最佳。由于存在可能无法导出的语法，不能用于MindIR导入导出。

        - **debug_level** (int) - 设置调试过程的配置。其值必须为 ``RELEASE`` 或 ``DEBUG`` 。默认值： ``RELEASE`` 。

          - ``RELEASE`` : 正常场景下使用，一些调试信息会被丢弃以获取一个较好的编译性能。
          - ``DEBUG`` : 当错误发生时，用来调试，在编译过程中，更多的调试信息会被记录下来。

        - **gpu_config** (dict) - 设置GPU硬件平台专用的参数，默认不设置。
          目前只支持GPU硬件平台上设置conv_fprop_algo、conv_dgrad_algo、conv_wgrad_algo、conv_allow_tf32和matmul_allow_tf32参数。

          - **conv_fprop_algo** (str): 指定Cudnn的卷积前向算法。默认值： ``normal`` 。其值范围如下：

            - normal:使用Cudnn自带的启发式搜索算法，会根据卷积形状和类型快速选择合适的卷积算法。该参数不保证性能最优。
            - performance: 使用Cudnn自带的试运行搜索算法，会根据卷积形状和类型试运行所有卷积算法，然后选择最优算法。该参数保证性能最优。
            - implicit_gemm: 该算法将卷积隐式转换成矩阵乘法，完成计算。不需要显式将输入张量数据转换成矩阵形式保存。
            - implicit_precomp_gemm: 该算法将卷积隐式转换成矩阵乘法，完成计算。但是需要一些额外的内存空间去保存预计算得到的索引值，以便隐式地将输入张量数据转换成矩阵形式。
            - gemm: 该算法将卷积显式转换成矩阵乘法，完成计算。在显式完成矩阵乘法过程中，需要额外申请内存空间，将输入转换成矩阵形式。
            - direct: 该算法直接完成卷积计算，不会隐式或显式的将卷积转换成矩阵乘法。
            - fft: 该算法利用快速傅里叶变换完成卷积计算。需要额外申请内存空间，保存中间结果。
            - fft_tiling: 该算法利用快速傅里叶变换完成卷积计算，但是需要对输入进行分块。同样需要额外申请内存空间，保存中间结果，但是对大尺寸的输入，所需内存空间小于 ``fft`` 算法。
            - winograd: 该算法利用Winograd变换完成卷积计算。需要额外申请内存空间，保存中间结果。
            - winograd_nonfused: 该算法利用Winograd变形算法完成卷积计算。需要额外申请内存空间，保存中间结果。
          - **conv_dgrad_algo** (str): 指定Cudnn的卷积输入数据的反向算法。默认值： ``normal`` 。其值范围如下：

            - normal:使用Cudnn自带的启发式搜索算法，会根据卷积形状和类型快速选择合适的卷积算法。该参数不保证性能最优。
            - performance: 使用Cudnn自带的试运行搜索算法，会根据卷积形状和类型试运行所有卷积算法，然后选择最优算法。该参数保证性能最优。
            - algo_0: 该算法将卷积表示为矩阵乘积的和，而没有实际显式地形成保存输入张量数据的矩阵。求和使用原子加法操作完成，因此结果是不确定的。
            - algo_1: 该算法将卷积表示为矩阵乘积，而没有实际显式地形成保存输入张量数据的矩阵。结果是确定的。
            - fft: 该算法利用快速傅里叶变换完成卷积计算。需要额外申请内存空间，保存中间结果。结果是确定的。
            - fft_tiling: 该算法利用快速傅里叶变换完成卷积计算，但是需要对输入进行分块。同样需要额外申请内存空间，保存中间结果，但是对大尺寸的输入，所需内存空间小于 ``fft`` 算法。结果是确定的。
            - winograd: 该算法利用Winograd变换完成卷积计算。需要额外申请内存空间，保存中间结果。结果是确定的。
            - winograd_nonfused: 该算法利用Winograd变形算法完成卷积计算。需要额外申请内存空间，保存中间结果。结果是确定的。

          - **conv_wgrad_algo** (str): 指定Cudnn的卷积输入卷积核的反向算法。默认值： ``normal`` 。其值范围如下：

            - normal:使用Cudnn自带的启发式搜索算法，会根据卷积形状和类型快速选择合适的卷积算法。该参数不保证性能最优。
            - performance: 使用Cudnn自带的试运行搜索算法，会根据卷积形状和类型试运行所有卷积算法，然后选择最优算法。该参数保证性能最优。
            - algo_0: 该算法将卷积表示为矩阵乘积的和，而没有实际显式地形成保存输入张量数据的矩阵。求和使用原子加法操作完成，因此结果是不确定的。
            - algo_1: 该算法将卷积表示为矩阵乘积，而没有实际显式地形成保存输入张量数据的矩阵。结果是确定的。
            - algo_3: 该算法类似于 ``algo_0`` ，但使用一些小的工作空间来预计算一些索引。结果也是不确定的。
            - fft: 该算法利用快速傅里叶变换完成卷积计算。需要额外申请内存空间，保存中间结果。结果是确定的。
            - fft_tiling: 该算法利用快速傅里叶变换完成卷积计算，但是需要对输入进行分块。同样需要额外申请内存空间，保存中间结果，但是对大尺寸的输入，所需内存空间小于 ``fft`` 算法。结果是确定的。
            - winograd_nonfused: 该算法利用Winograd变形算法完成卷积计算。需要额外申请内存空间，保存中间结果。结果是确定的。

          - **conv_allow_tf32** (bool): 该标志表示是否开启卷积在CUDNN下的TF32张量核计算。默认值： ``True`` 。

          - **matmul_allow_tf32** (bool): 该标志表示是否开启矩阵乘在CUBLAS下的TF32张量核计算。默认值： ``False`` 。

        - **jit_config** (dict) - 设置全局编译选项的配置，只在使用Cell或者jit装饰器定义的网络中生效，默认不设置。
          context设置全局jit config，而JitConfig设置局部网络的jit config，二者同时存在时，全局jit config不会覆盖局部网络的jit config。

          - **jit_level** (str): 用来控制编译优化级别。默认值为空，框架根据产品类别自动选择优化级别，Altas训练产品为O2，其余产品均为O0。其值范围如下：

            - O0: 除必要影响功能的优化外，其他优化均关闭，使用逐算子执行的执行方式。
            - O1: 使能常用优化和自动算子融合优化，使用逐算子执行的执行方式。
            - O2: 开启极致性能优化，使用下沉的执行方式。

          - **infer_boost** (str): 用来使能推理模式。默认值为“off”，表示关闭。其值范围如下：

            - on: 开启推理模式，推理性能得到较大提升。
            - off: 关闭推理模式，使用前向运算进行推理，性能较差。

    异常：
        - **ValueError** - 输入key不是上下文中的属性。
