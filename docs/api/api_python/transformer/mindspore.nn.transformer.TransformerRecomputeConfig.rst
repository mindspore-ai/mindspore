.. py:class:: mindspore.nn.transformer.TransformerRecomputeConfig(recompute=False, parallel_optimizer_comm_recompute=False, mp_comm_recompute=True, recompute_slice_activation=False)

    Transformer的重计算配置接口。

    参数：
        - **recompute** (bool) - 是否使能重计算。默认值为False。
        - **parallel_optimizer_comm_recompute** (bool) - 指定由优化器切分产生的AllGather算子是否进行重计算。默认值为False。
        - **mp_comm_recompute** (bool) - 指定由模型并行成分产生的通信算子是否进行重计算。默认值为True。
        - **recompute_slice_activation** (bool) - 指定激活层是否切片保存。默认值为False。
