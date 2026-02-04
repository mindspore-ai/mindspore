mindspore.dataset.dataloader.RandomSampler
==========================================

.. py:class:: mindspore.dataset.dataloader.RandomSampler(data_source, replacement=False, num_samples=None, generator=None)

    对数据集进行随机采样。

    参数：
        - **data_source** (Dataset) - 要从中加载数据的数据集。
        - **replacement** (bool, 可选) - 是否开启放回采样。默认值： ``False`` 。
        - **num_samples** (Union[int, None], 可选) - 要抽取的样本数量。默认值： ``None`` ，将会设置为 `data_source` 的长度。
        - **generator** (numpy.random.Generator, 可选) - 采样时使用的生成器。默认值： ``None`` 。
