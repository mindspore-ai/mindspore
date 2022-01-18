mindspore.nn.DistributedGradReducer
===================================

.. py:class:: mindspore.nn.DistributedGradReducer(parameters, mean=True, degree=None, fusion_type=1, group=GlobalComm.WORLD_COMM_GROUP)

    分布式优化器。

    用于数据并行模式中，对所有卡的梯度利用AllReduce进行聚合。

    **参数：**

    - **parameters** (list) - 需要更新的参数。
    - **mean** (bool) - 当mean为True时，对AllReduce之后的梯度求均值。默认值：False。
    - **degree** (int) - 平均系数，通常等于设备编号。默认值：None。
    - **fusion_type** (int) - AllReduce算子的融合类型。默认值：1。

    **异常：**

    **ValueError**：如果degree不是int或小于0。
