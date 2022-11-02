mindspore.dataset.vision.RandomSelectSubpolicy
==============================================

.. py:class:: mindspore.dataset.vision.RandomSelectSubpolicy(policy)

    从策略列表中随机选择一个子策略以应用于输入图像。

    参数：
        - **policy** (list[list[tuple[TensorOperation, float]]]) - 可供选择的子策略列表。子策略是一系列 (operation, prob) 格式的元组组成的列表，其中 `operation` 是数据处理操作， `prob` 是应用此操作的概率， `prob` 值必须在 [0.0, 1.0] 范围内。一旦选择了子策略，子策略中的每个操作都将根据其概率依次应用。

    异常：
        - **TypeError** - 当 `policy` 包含无效数据处理操作。
