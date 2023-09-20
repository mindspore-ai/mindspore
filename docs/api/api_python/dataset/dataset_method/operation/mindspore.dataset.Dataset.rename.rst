mindspore.dataset.Dataset.rename
================================

.. py:method:: mindspore.dataset.Dataset.rename(input_columns, output_columns)

    对数据集对象按指定的列名进行重命名。

    参数：
        - **input_columns** (Union[str, list[str]]) - 待重命名的列名列表。
        - **output_columns** (Union[str, list[str]]) - 重命名后的列名列表。

    返回：
        Dataset，应用了上述操作的新数据集对象。
