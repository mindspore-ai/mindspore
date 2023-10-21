mindspore.dataset.Dataset.split
===============================

.. py:method:: mindspore.dataset.Dataset.split(sizes, randomize=True)

    将数据集拆分为多个不重叠的子数据集。

    参数：
        - **sizes** (Union[list[int], list[float]]) - 如果指定了一列整数[s1, s2, …, sn]，数据集将被拆分为n个大小为s1、s2、...、sn的数据集。如果所有输入大小的总和不等于原始数据集大小，则报错。如果指定了一列浮点数[f1, f2, …, fn]，则所有浮点数必须介于0和1之间，并且总和必须为1，否则报错。数据集将被拆分为n个大小为round(f1*K)、round(f2*K)、...、round(fn*K)的数据集，其中K是原始数据集的大小。
          如果round四舍五入计算后：

          - 任何子数据集的的大小等于0，都将发生错误。
          - 如果子数据集大小的总和小于K，K - sigma(round(fi * k))的值将添加到第一个子数据集，sigma为求和操作。
          - 如果子数据集大小的总和大于K，sigma(round(fi * K)) - K的值将从第一个足够大的子数据集中删除，且删除后的子数据集大小至少大于1。

        - **randomize** (bool, 可选) - 确定是否随机拆分数据。默认值： ``True`` ，数据集将被随机拆分。否则将按顺序拆分为多个不重叠的子数据集。

    .. note::
        1. 如果进行拆分操作的数据集对象为MappableDataset类型，则将自动调用一个优化后的split操作。
        2. 如果进行split操作，则不应对数据集对象进行分片操作（如指定num_shards或使用 :class:`mindspore.dataset.DistributedSampler` ）。相反，如果创建一个 :class:`mindspore.dataset.DistributedSampler` ，并在split操作拆分后的子数据集对象上进行分片操作，强烈建议在每个子数据集上设置相同的种子，否则每个分片可能不是同一个子数据集的一部分（请参见示例）。
        3. 强烈建议不要对数据集进行混洗，而是使用随机化（ `randomize` 为 ``True`` ）。对数据集进行混洗的结果具有不确定性，每个拆分后的子数据集中的数据在每个epoch可能都不同。

    异常：
        - **RuntimeError** - 数据集对象不支持 `get_dataset_size` 或者 `get_dataset_size` 返回None。
        - **RuntimeError** - `sizes` 是list[int]，并且 `sizes` 中所有元素的总和不等于数据集大小。
        - **RuntimeError** - `sizes` 是list[float]，并且计算后存在大小为0的拆分子数据集。
        - **RuntimeError** - 数据集对象在调用拆分之前已进行分片。
        - **ValueError** - `sizes` 是list[float]，且并非所有float数值都在0和1之间，或者float数值的总和不等于1。

    返回：
        Tuple[Dataset]，从原数据集拆分出的新数据集构成的元组。
