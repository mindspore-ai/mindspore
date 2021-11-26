mindspore.dataset.concat
=========================

.. py:method:: concat(datasets)

    对传入的多个数据集对象进行拼接操作。重载“+”运算符来进行数据集对象拼接操作。

    .. note::用于拼接的多个数据集对象，其列名、每列数据的维度（rank)和类型必须相同。

    **参数：**

    **datasets** (Union[list, class Dataset])：与当前数据集对象拼接的数据集对象列表或单个数据集对象。


    **返回：**

    ConcatDataset，拼接后的数据集对象。

    **样例：**

    >>> # 通过使用“+”运算符拼接dataset_1和dataset_2，获得拼接后的数据集对象
    >>> dataset = dataset_1 + dataset_2
    >>> # 通过concat操作拼接dataset_1和dataset_2，获得拼接后的数据集对象
    >>> dataset = dataset_1.concat(dataset_2)