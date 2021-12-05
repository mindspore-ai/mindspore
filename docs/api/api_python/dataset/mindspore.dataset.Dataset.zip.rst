    .. py:method:: zip(datasets)

        将数据集和输入的数据集或者数据集元组按列进行合并压缩。输入数据集中的列名必须不同。

        **参数：**

        - **datasets** (Union[tuple, class Dataset]) - 数据集对象的元组或单个数据集对象与当前数据集一起合并压缩。

        **返回：**

        ZipDataset，合并压缩后的数据集对象。

        **样例：**

        >>> # 创建一个数据集，它将dataset和dataset_1进行合并
        >>> dataset = dataset.zip(dataset_1)