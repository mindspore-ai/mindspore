mindspore.set_auto_parallel_context
====================================


.. py:function:: mindspore.set_auto_parallel_context(**kwargs)

    配置自动并行，当前CPU仅支持数据并行。

    .. note::
        配置时，必须输入配置的名称。如果某个程序具有不同并行模式下的任务，需要提前调用 :func:`mindspore.reset_auto_parallel_context` 为下一个任务设置新的并行模式。若要设置或更改并行模式，必须在创建任何Initializer之前调用接口，否则，在编译网络时，可能会出现RuntimeError。

    某些配置适用于特定的并行模式，有关详细信息，请参见下表：

    =========================  =========================
             Common                  AUTO_PARALLEL
    =========================  =========================
    device_num                 gradient_fp32_sync
    global_rank                loss_repeated_mean
    gradients_mean             auto_parallel_search_mode
    parallel_mode              strategy_ckpt_load_file
    all_reduce_fusion_config   strategy_ckpt_save_file
    enable_parallel_optimizer  dataset_strategy
    enable_alltoall            pipeline_stages
    \                          grad_accumulation_step
    \                          strategy_ckpt_config
    =========================  =========================

    参数：
        - **device_num** (int) - 表示可用设备的编号，必须在[1,4096]范围中。默认值：1。
        - **global_rank** (int) - 表示全局RANK的ID，必须在[0,4095]范围中。默认值：0。
        - **gradients_mean** (bool) - 表示是否在梯度的 AllReduce后执行平均算子。stand_alone不支持gradients_mean。默认值：False。
        - **gradient_fp32_sync** (bool) - 在FP32中运行梯度的 AllReduce。stand_alone、data_parallel和hybrid_parallel不支持gradient_fp32_sync。默认值：True。
        - **parallel_mode** (str) - 有五种并行模式，分别是stand_alone、data_parallel、hybrid_parallel、semi_auto_parallel和auto_parallel。默认值：stand_alone。

          - stand_alone：单卡模式。
          - data_parallel：数据并行模式。
          - hybrid_parallel：手动实现数据并行和模型并行。
          - semi_auto_parallel：半自动并行模式。
          - auto_parallel：自动并行模式。

        - **search_mode** (str) - 表示有三种策略搜索模式，分别是recursive_programming，dynamic_programming和sharding_propagation。默认值：dynamic_programming。

          - recursive_programming：表示双递归搜索模式。
          - dynamic_programming：表示动态规划搜索模式。
          - sharding_propagation：表示从已配置算子的切分策略传播到所有算子。

        - **auto_parallel_search_mode** (str) - search_mode参数的兼容接口。将在后续的版本中删除。
        - **parameter_broadcast** (bool) - 表示在训练前是否广播参数。在训练之前，为了使所有设备的网络初始化参数值相同，请将设备0上的参数广播到其他设备。不同并行模式下的参数广播不同。在data_parallel模式下，除layerwise_parallel属性为True的参数外，所有参数都会被广播。在hybrid_parallel、semi_auto_parallel和auto_parallel模式下，分段参数不参与广播。默认值：False。
        - **strategy_ckpt_load_file** (str) - 表示用于加载并行策略checkpoint的路径。默认值： ''。
        - **strategy_ckpt_save_file** (str) - 表示用于保存并行策略checkpoint的路径。默认值： ''。
        - **full_batch** (bool) - 如果在auto_parallel模式下加载整个batch数据集，则此参数应设置为True。默认值：False。目前不建议使用该接口，建议使用dataset_strategy来替换它。
        - **dataset_strategy** (Union[str, tuple]) - 表示数据集分片策略。默认值：data_parallel。dataset_strategy="data_parallel"等于full_batch=False，dataset_strategy="full_batch"等于full_batch=True。对于在静态图模式下执行并且通过模型并列策略加载到网络的数据集，如ds_stra ((1, 8)、(1, 8))，需要使用set_auto_parallel_context(dataset_strategy=ds_stra)。
        - **enable_parallel_optimizer** (bool) - 这是一个开发中的特性，它可以为数据并行训练对权重更新计算进行分片，以节省时间和内存。目前，自动和半自动并行模式支持Ascend和GPU中的所有优化器。数据并行模式仅支持Ascend中的 `Lamb` 和 `AdamWeightDecay` 。默认值：False。
        - **enable_alltoall** (bool) - 允许在通信期间生成 `AllToAll` 通信算子的开关。如果其值为 False，则将由 `AllGather` 、 `Split` 和 `Concat` 等通信算子的组合来代替 `AllToAll` 。默认值：False。
        - **all_reduce_fusion_config** (list) - 通过参数索引设置 AllReduce 融合策略。仅支持ReduceOp.SUM和HCCL_WORLD_GROUP/NCCL_WORLD_GROUP。没有默认值。如果不设置，则关闭算子融合。
        - **pipeline_stages** (int) - 设置pipeline并行的阶段信息。这表明了设备如何单独分布在pipeline上。所有的设备将被划分为pipeline_stags个阶段。目前，这只能在启动semi_auto_parallel模式的情况下使用。默认值：1。
        - **grad_accumulation_step** (int) - 在自动和半自动并行模式下设置梯度的累积step。其值应为正整数。默认值：1。
        - **parallel_optimizer_config** (dict) - 用于开启优化器并行后的行为配置。仅在enable_parallel_optimizer=True的时候生效。目前，它支持关键字如下的关键字：

          - gradient_accumulation_shard(bool)：设置累积梯度变量是否在数据并行维度上进行切分。开启后，将进一步减小模型的显存占用，但是会在反向计算梯度时引入额外的通信算子（ReduceScatter）。此配置仅在流水线并行训练和梯度累积模式下生效。默认值：True。
          - parallel_optimizer_threshold(int)：设置参数切分的阈值。占用内存小于该阈值的参数不做切分。占用内存大小 = shape[0] \* ... \* shape[n] \* size(dtype)。该阈值非负。单位：KB。默认值：64。

        - **comm_fusion** (dict) - 用于设置通信算子的融合配置。可以同一类型的通信算子按梯度张量的大小或者顺序分块传输。输入格式为{"通信类型": {"mode":str, "config": None int 或者 list}},每种通信算子的融合配置有两个键："mode"和"config"。支持以下通信类型的融合类型和配置：

          - openstate：是否开启通信融合功能。通过 True 或 False 来开启或关闭通信融合功能。默认值：True。
          - allreduce：进行AllReduce算子的通信融合。"mode"包含："auto"、"size"和"index"。在"auto"模式下，融合的是梯度变量的大小，默认值阈值为"64"MB，"config"对应的值为None。在"size"模式下，需要用户在config的字典中指定梯度大小阈值，这个值必须大于"0"MB。在"mode"为"index"时，它与"all_reduce_fusion_config"相同，用户需要给"config"传入一个列表，里面每个值表示梯度的索引。
          - allgather：进行AllGather算子的通信融合。"mode"包含："auto"、"size"。"auto" 和 "size"模式的配置方式与AllReduce相同。
          - reducescatter：进行ReduceScatter算子的通信融合。"mode"包含："auto"、"size"。"auto" 和 "size"模式的配置方式与AllReduce相同。

        - **strategy_ckpt_config** (dict) - 用于设置并行策略文件的配置。包含 `strategy_ckpt_load_file` 和 `strategy_ckpt_save_file` 两个参数的功能，建议使用此参数替换这两个参数。它包含以下配置：

          - load_file(str)：加载并行切分策略的路径。如果文件扩展名为 `.json`，文件以json格式加载。否则，文件以ProtoBuf格式加载。默认值：""。
          - save_file(str)：保存并行切分策略的路径。如果文件扩展名为 `.json`，文件以json格式保存。否则，文件以ProtoBuf格式保存。默认值：""。
          - only_trainable_params(bool)：仅保存/加载可训练参数的策略信息。默认值：True。

    异常：
        - **ValueError** - 输入key不是自动并行上下文中的属性。
