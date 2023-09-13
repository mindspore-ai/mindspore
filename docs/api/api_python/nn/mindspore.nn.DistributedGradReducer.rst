mindspore.nn.DistributedGradReducer
===================================

.. py:class:: mindspore.nn.DistributedGradReducer(parameters, mean=None, degree=None, fusion_type=1, group=GlobalComm.WORLD_COMM_GROUP)

    分布式优化器。

    用于数据并行模式中，对所有卡的梯度利用AllReduce进行聚合。

    参数：
        - **parameters** (list) - 需要更新的参数。
        - **mean** (bool) - 当mean为True时，对AllReduce之后的梯度求均值。未指定时，使用auto_parallel_context中的配置“gradients_mean”。 默认值： ``None`` 。
        - **degree** (int) - 平均系数，通常等于设备编号。默认值： ``None`` 。
        - **fusion_type** (int) - AllReduce算子的融合类型。默认值： ``1`` 。
        - **group** (str) - AllReduce算子的通信域，若需要自定义通信域，需要调用create_group接口。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    异常：
        - **ValueError** - 如果degree不是int或小于0。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。