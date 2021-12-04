    .. py:method:: add_sampler(new_sampler)

        为当前数据集添加采样器。

        **参数：**

        - **new_sampler** (Sampler) ：作用于当前数据集的采样器。

        **样例：**

        >>> # dataset为任意数据集实例
        >>> # 对该数据集应用DistributedSampler
        >>> new_sampler = ds.DistributedSampler(10, 2)
        >>> dataset.add_sampler(new_sampler)