.. py:method:: add_child(sampler)

    为给定采样器添加子采样器。子采样器接收父采样器输出数据作为输入，并应用其采样逻辑返回新的采样结果。

    参数：
        - **sampler** (Sampler) - 用于从数据集中选择样本的对象。仅支持内置采样器（DistributedSampler、PKSampler、RandomSampler、SequentialSampler、SubsetRandomSampler、WeightedRandomSampler）。

.. py:method:: get_child()

    获取给定采样器的子采样器。

    返回：
        Sampler，给定采样器的子采样器。
