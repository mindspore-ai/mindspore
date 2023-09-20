mindspore.dataset.Dataset.project
=================================

.. py:method:: mindspore.dataset.Dataset.project(columns)

    从数据集对象中选择需要的列，并按给定的列名的顺序进行排序。
    未指定的数据列将被丢弃。

    参数：
        - **columns** (Union[str, list[str]]) - 要选择的数据列的列名列表。

    返回：
        Dataset，应用了上述操作的新数据集对象。
