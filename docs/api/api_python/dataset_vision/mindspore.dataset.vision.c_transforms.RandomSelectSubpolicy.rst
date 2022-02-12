mindspore.dataset.vision.c_transforms.RandomSelectSubpolicy
===========================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomSelectSubpolicy(policy)

    从策略列表中选择一个随机子策略以应用于输入图像。

    **参数：**

    - **policy** (list(list(tuple(TensorOp, prob (float))))) - 可供选择的子策略列表。 子策略是一系列 (op, prob) 格式的元组组成的列表，其中 `op` 是针对 Tensor 的操作， `prob` 是应用此操作的概率， `prob` 值必须在 [0, 1] 范围内。 一旦选择了子策略，子策略中的每个操作都将根据其概率依次应用。
