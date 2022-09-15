mindspore.dataset.Dataset.concat
================================

.. py:method:: mindspore.dataset.Dataset.concat(datasets)

    对传入的多个数据集对象进行拼接操作。可以使用"+"运算符来进行数据集进行拼接。

    .. note::
        用于拼接的多个数据集对象，每个数据集对象的列名、每列数据的维度（rank）和数据类型必须相同。

    参数：
        - **datasets** (Union[list, Dataset]) - 与当前数据集对象拼接的数据集对象列表或单个数据集对象。

    返回：
        Dataset，拼接后的数据集对象。
