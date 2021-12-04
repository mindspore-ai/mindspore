    .. py:method:: use_sampler(new_sampler)

        为当前数据集更换一个新的采样器。

        **参数：**

        - **new_sampler** (Sampler) ：替换的新采样器。

        **样例：**

        >>> # dataset为任意数据集实例
        >>> # 将该数据集的采样器更换为DistributedSampler
        >>> new_sampler = ds.DistributedSampler(10, 2)
        >>> dataset.use_sampler(new_sampler)