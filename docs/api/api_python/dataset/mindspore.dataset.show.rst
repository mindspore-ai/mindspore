mindspore.dataset.show
======================

.. py:method:: mindspore.dataset.show(dataset, indentation=2)

    将数据处理管道图写入MindSpore的INFO级别日志文件。

    **参数：**

    - **dataset** (Dataset): 数据处理管道对象。
    - **indentation** (int, optional): 设置MindSpore的INFO级别日志文件打印时的缩进字符数。若为None，则不缩进。

    **样例：**

    >>> dataset = ds.MnistDataset(mnist_dataset_dir, 100)
    >>> one_hot_encode = c_transforms.OneHot(10)
    >>> dataset = dataset.map(operation=one_hot_encode, input_column_names="label")
    >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
    >>> ds.show(dataset)
        