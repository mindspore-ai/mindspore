mindspore.dataset.Dataset.concat
================================

.. py:method:: mindspore.dataset.Dataset.concat(datasets)

    对传入的多个数据集对象进行拼接操作。可以使用"+"运算符对数据集进行拼接。

    对于由多个数据集对象拼接而成的数据集，返回的数据按照传入数据集的顺序排列。如需改变数据顺序（例如从每个数据集中随机选择而非按顺序），可在拼接后的数据集对象上应用 `use_sampler` 方法。当前 `use_sampler` 支持 `dataset.DistributedSampler` （用于从每个数据集中进行分片选择）或 `dataset.RandomSampler` （用于从每个数据集中进行随机选择）。

    .. note::
        用于拼接的多个数据集对象，每个数据集对象的列名、每列数据的维度（rank）和数据类型必须相同。

    参数：
        - **datasets** (Union[list, Dataset]) - 与当前数据集对象拼接的数据集对象列表或单个数据集对象。

    返回：
        Dataset，应用了上述操作的新数据集对象。
