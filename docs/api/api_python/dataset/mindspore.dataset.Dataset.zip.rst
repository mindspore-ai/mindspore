    .. py:method:: zip(datasets)

        将数据集对象和输入的数据集对象或者数据集对象元组按列进行合并压缩。输入数据集对象中不能有重名的列。

        **参数：**

        - **datasets** (Union[tuple, class Dataset]) - 数据集对象的元组或单个数据集对象与当前数据集对象一起合并压缩。

        **返回：**

        ZipDataset，合并压缩后的数据集对象。
