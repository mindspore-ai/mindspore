.. py:method:: project(columns)

    从数据集对象中选择需要的列，并按给定的列名的顺序进行排序。
    未指定的数据列将被丢弃。

    参数：
        - **columns** (Union[str, list[str]]) - 要选择的数据列的列名列表。

    返回：
        Dataset，project操作后的数据集对象。

.. py:method:: rename(input_columns, output_columns)

    对数据集对象按指定的列名进行重命名。

    参数：
        - **input_columns** (Union[str, list[str]]) - 待重命名的列名列表。
        - **output_columns** (Union[str, list[str]]) - 重命名后的列名列表。

    返回：
        RenameDataset，rename操作后的数据集对象。

.. py:method:: repeat(count=None)

    重复此数据集 `count` 次。如果 `count` 为None或-1，则无限重复迭代。

    .. note::
        repeat和batch的顺序反映了batch的数量。建议：repeat操作在batch操作之后使用。

    参数：
        - **count** (int) - 数据集重复的次数。默认值：None。

    返回：
        RepeatDataset，repeat操作后的数据集对象。

.. py:method:: reset()

    重置下一个epoch的数据集对象。

