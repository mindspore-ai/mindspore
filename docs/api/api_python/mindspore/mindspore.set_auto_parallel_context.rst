mindspore.set_auto_parallel_context
====================================


.. py:function:: mindspore.set_auto_parallel_context(**kwargs)

    配置自动并行，此接口将在后续版本中废弃，请使用接口 :class:`mindspore.parallel.auto_parallel.AutoParallel` 代替。

    .. note::
        当前CPU仅支持数据并行。

    某些配置适用于特定的并行模式，有关详细信息，请参见下表：

    =========================  =========================
             Common                  AUTO_PARALLEL
    =========================  =========================
    device_num                   gradient_fp32_sync
    global_rank                  loss_repeated_mean
    gradients_mean               search_mode
    parallel_mode                parameter_broadcast
    all_reduce_fusion_config     strategy_ckpt_load_file
    enable_parallel_optimizer    strategy_ckpt_save_file
    parallel_optimizer_config    dataset_strategy
    enable_alltoall              pipeline_stages
    pipeline_config              auto_parallel_search_mode
    force_fp32_communication     pipeline_result_broadcast
               \                 comm_fusion
               \                 strategy_ckpt_config
               \                 group_ckpt_save_file
               \                 auto_pipeline
               \                 dump_local_norm
               \                 dump_local_norm_path
               \                 dump_device_local_norm
    =========================  =========================

    参数：
        - **device_num** (int) - 表示可用设备的编号，必须在[1,4096]范围中。默认值： ``1`` 。
        - **global_rank** (int) - 表示全局RANK的ID，必须在[0,4095]范围中。默认值： ``0`` 。
        - **gradients_mean** (bool) - 表示是否在梯度的 AllReduce后执行平均算子。stand_alone不支持gradients_mean。默认值： ``False`` 。
        - **gradient_fp32_sync** (bool) - 在FP32中运行梯度的 AllReduce。stand_alone、data_parallel和hybrid_parallel不支持gradient_fp32_sync。默认值： ``True`` 。
        - **loss_repeated_mean** (bool) - 表示在重复计算时，是否向后执行均值操作符。默认值： ``True`` 。
        - **parallel_mode** (str) - 有五种并行模式，分别是 ``stand_alone`` 、 ``data_parallel`` 、 ``hybrid_parallel`` 、 ``semi_auto_parallel`` 和 ``auto_parallel`` 。默认值： ``stand_alone`` 。

          - stand_alone：单卡模式。
          - data_parallel：数据并行模式。
          - hybrid_parallel：手动实现数据并行和模型并行。
          - semi_auto_parallel：半自动并行模式。
          - auto_parallel：自动并行模式。

        - **search_mode** (str) - 表示有三种策略搜索模式，分别是 ``recursive_programming`` 、 ``sharding_propagation`` 和 ``dynamic_programming`` (不推荐使用)。只有在 ``auto_parallel`` 模式下生效。默认值： ``recursive_programming`` 。

          - recursive_programming：表示双递归搜索模式。为了获取最优性能，建议用户设置batch size大于等于设备数与多副本并行数的乘积。
          - sharding_propagation：表示从已配置算子的切分策略传播到所有算子。目前不支持动态shape。
          - dynamic_programming：表示动态规划搜索模式。

        - **auto_parallel_search_mode** (str) - search_mode参数的兼容接口。将在后续的版本中删除。
        - **parameter_broadcast** (bool) - 表示在训练前是否广播参数。在训练之前，为了使所有设备的网络初始化参数值相同，请将设备0上的参数广播到其他设备。不同并行模式下的参数广播不同。在 ``data_parallel`` 模式下，除layerwise_parallel属性为 ``True`` 的参数外，所有参数都会被广播。在 ``hybrid_parallel`` 、 ``semi_auto_parallel`` 和 ``auto_parallel`` 模式下，分段参数不参与广播。默认值： ``False`` 。
        - **strategy_ckpt_load_file** (str) - 表示用于加载并行策略checkpoint的路径。目前不建议使用该参数，建议使用 `strategy_ckpt_config` 来替代它。默认值： ``''`` 。
        - **strategy_ckpt_save_file** (str) - 表示用于保存并行策略checkpoint的路径。目前不建议使用该参数，建议使用 `strategy_ckpt_config` 来替代它。默认值： ``''`` 。
        - **full_batch** (bool) - 如果在 ``auto_parallel`` 模式下加载整个batch数据集，则此参数应设置为 ``True`` 。默认值： ``False`` 。目前不建议使用该接口，建议使用dataset_strategy来替换它。
        - **dataset_strategy** (Union[str, tuple]) - 表示数据集分片策略。默认值： ``data_parallel`` 。dataset_strategy="data_parallel"等价于full_batch=False，dataset_strategy="full_batch"等价于full_batch=True。对于在静态图模式下执行并且通过模型并列策略加载到网络的数据集分片策略，如ds_stra ((1, 8)、(1, 8))，需要使用set_auto_parallel_context(dataset_strategy=ds_stra)。数据集分片策略不受当前配置的并行模式影响。dataset strategy同时也支持配置元组，元组中每个元素都是Layout。
        - **enable_parallel_optimizer** (bool) - 这是一个开发中的特性，它可以为数据并行训练对权重更新计算进行分片，以节省时间和内存。目前，自动和半自动并行模式支持Ascend和GPU中的所有优化器。数据并行模式仅支持Ascend中的 `Lamb` 和 `AdamWeightDecay` 。默认值： ``False`` 。
        - **enable_alltoall** (bool) - 允许在通信期间生成 `AllToAll` 通信算子的开关。如果其值为 False，则将由 `AllGather` 、 `Split` 和 `Concat` 等通信算子的组合来代替 `AllToAll` 。默认值： ``False`` 。
        - **force_fp32_communication** (bool) - 通信期间reduce类算子（AllReduce、ReduceScatter）是否强制使用fp32数据类型进行通信的开关。True为开启开关。默认值： ``False`` 。
        - **all_reduce_fusion_config** (list) - 通过参数索引设置 AllReduce 融合策略。仅支持ReduceOp.SUM和HCCL_WORLD_GROUP/NCCL_WORLD_GROUP。没有默认值。如果不设置，则关闭算子融合。
        - **pipeline_stages** (int) - 设置pipeline并行的阶段信息。表明设备如何单独分布在pipeline上。所有的设备将被划分为pipeline_stages个阶段。默认值： ``1`` 。
        - **pipeline_result_broadcast** (bool) - 表示pipeline并行推理时，最后一个stage的结果是否广播给其余stage。默认值： ``False`` 。
        - **auto_pipeline** (bool) - 自动设置流水线阶段数。其值将在1和输入的 `pipeline_stages` 之间选择。本功能需要将 `parallel_mode` 设置成自动并行 ``auto_parallel`` 并将 `search_mode` 设置成双递归算法 ``recursive_programming``。默认值： ``False`` 。
        - **pipeline_config** (dict) - 用于设置开启pipeline并行后的行为配置。目前支持如下关键字：

          - pipeline_interleave(bool)：表示是否开启interleave。
          - pipeline_scheduler(str)：表示pipeline并行使用的调度策略。当前仅支持 ``gpipe/1f1b/seqpipe/seqvpp/seqsmartvpp/zero_bubble_v``。当应用seqsmartvpp时，流水线并行必须是偶数。
        - **parallel_optimizer_config** (dict) - 用于开启优化器并行后的行为配置。仅在enable_parallel_optimizer=True的时候生效。目前支持如下关键字：

          - gradient_accumulation_shard(bool)：请使用 optimizer_level: ``level2`` 替换此配置。设置累加梯度变量是否在数据并行维度上进行切分。开启后，将进一步减小模型的显存占用，但是会在反向计算梯度时引入额外的通信算子（ReduceScatter）。此配置仅在流水线并行训练和梯度累加模式下生效。默认值： ``True`` 。
          - parallel_optimizer_threshold(int)：设置参数切分的阈值。占用内存小于该阈值的参数不做切分。占用内存大小 = shape[0] \* ... \* shape[n] \* size(dtype)。该阈值非负。单位：KB。默认值： ``64`` 。
          - optimizer_weight_shard_size(int)：设置指定优化器权重切分通信域的大小。只有当启用优化器并行时生效。数值范围可以是(0, device_num]或者-1，若同时开启流水线并行，数值范围则为(0, device_num/stage]或者-1。如果参数的数据并行通信域大小不能被 `optimizer_weight_shard_size` 整除，那么指定的优化器权重切分通信域大小就不会生效。默认值为 ``-1`` ，表示优化器权重切片通信域大小是每个参数的数据并行通信域大小。
          - optimizer_level(str，可选): optimizer_level配置用于指定优化器切分的切分级别。需要注意的是，静态图的优化器并行实现与动态图比如megatron不一致，但是显存优化效果相同。当 optimizer_level= ``level1`` 时，对权重与优化器状态进行切分。optimizer_level= ``level2`` 时，对权重、优化器状态以及梯度进行切分。当optimizer_level= ``level3`` 时，对权重、优化器状态、梯度进行切分，并且在反向开始前会对权重额外展开一次allgather通信，以释放前向allgather的显存。它必须是[ ``level1`` 、 ``level2`` 、 ``level3`` ]中的一个。默认值: ``level1``。

        - **comm_fusion** (dict) - 用于设置通信算子的融合配置。可以同一类型的通信算子按梯度张量的大小或者顺序分块传输。输入格式为{"通信类型": {"mode":str, "config": None int 或者 list}}，每种通信算子的融合配置有两个键："mode"和"config"。支持以下通信类型的融合类型和配置：

          - openstate：是否开启通信融合功能。通过 ``True`` 或 ``False`` 来开启或关闭通信融合功能。默认值： ``True`` 。
          - allreduce：进行AllReduce算子的通信融合。"mode"包含"auto"、"size"和"index"。在"auto"模式下，融合梯度变量的大小，默认值阈值为64MB，"config"对应的值为None。在"size"模式下，需要用户在config的字典中指定梯度大小阈值，这个值必须大于"0"MB。在"mode"为"index"时，它与"all_reduce_fusion_config"相同，用户需要给"config"传入一个列表，里面每个值表示梯度的索引。
          - allgather：进行AllGather算子的通信融合。"mode"包含"auto"、"size"。"auto" 和 "size"模式的配置方式与AllReduce相同。
          - reducescatter：进行ReduceScatter算子的通信融合。"mode"包含"auto"、"size"。"auto" 和 "size"模式的配置方式与AllReduce相同。

        - **strategy_ckpt_config** (dict) - 用于设置并行策略文件的配置。包含 `strategy_ckpt_load_file` 和 `strategy_ckpt_save_file` 两个参数的功能，建议使用此参数替代这两个参数。它包含以下配置：

          - load_file(str)：加载并行切分策略的路径。如果文件扩展名为 `.json`，文件以json格式加载。否则，文件以ProtoBuf格式加载。默认值： ``""``。
          - save_file(str)：保存并行切分策略的路径。如果文件扩展名为 `.json`，文件以json格式保存。否则，文件以ProtoBuf格式保存。默认值： ``""``。
          - only_trainable_params(bool)：仅保存/加载可训练参数的策略信息。默认值： ``True`` 。
        - **group_ckpt_save_file** (str) - 在自动/半自动并行场景下，指定图编译过程中所创建group的保存路径。
        - **dump_local_norm** (bool) - 在自动/半自动并行场景下，指定是否打印local norm值。
        - **dump_local_norm_path** (str) - 在自动/半自动并行场景下，指定local norm值的保存路径。默认值： ``""``。
        - **dump_device_local_norm** (bool) - 在自动/半自动并行场景下，指定是否打印device local norm值。

    异常：
        - **ValueError** - 输入key不是自动并行上下文中的属性。
