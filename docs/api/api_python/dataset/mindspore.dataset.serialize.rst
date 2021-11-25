mindspore.dataset.serialize
============================

.. py:method:: mindspore.dataset.serialize(dataset, json_filepath='')

    将数据处理管道序列化成JSON文件。

    .. note::
        目前不支持某些Python对象序列化。对于map算子的自定义Python函数序列化， `mindspore.dataset.serialize` 仅返回其函数名称。

    **参数：**

    - **dataset** (Dataset): 数据处理管道对象。
    - **json_filepath** (str): 生成序列化JSON文件的路径。

    **返回：**

    Dict，包含序列化数据集图的字典。

    **异常：**

    **OSError:** 无法打开文件。

    **样例：**

    >>> dataset = ds.MnistDataset(mnist_dataset_dir, 100)
    >>> one_hot_encode = c_transforms.OneHot(10)  # num_classes是输入参数
    >>> dataset = dataset.map(operation=one_hot_encode, input_column_names="label")
    >>> dataset = dataset.batch(batch_size=10, drop_remainder=True)
    >>> # 将其序列化为JSON文件
    >>> ds.engine.serialize(dataset, json_filepath="/path/to/mnist_dataset_pipeline.json")
    >>> serialized_data = ds.engine.serialize(dataset)  # 将其序列化为Python字典
    