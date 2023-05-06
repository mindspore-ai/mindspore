mindspore.dataset.serialize
============================

.. py:function:: mindspore.dataset.serialize(dataset, json_filepath='')

    将数据处理管道序列化成JSON文件。

    .. note::
        目前不支持对Python对象进行完整的序列化。不支持的场景包括数据管道中使用了 `GeneratorDataset` 或包含自定义Python函数的 `map` 或 `batch` 操作。
        对于Python对象，序列化操作不会得到完整的对象内容，这意味着对序列化得到的JSON文件进行反序列化时，可能会导致错误。
        例如，对自定义Python函数（Python user-defined functions）的数据管道进行序列化时，会出现相关警告提示，并且得到的JSON文件不能被反序列化为可用的数据管道。

    参数：
        - **dataset** (Dataset) - 数据处理管道对象。
        - **json_filepath** (str) - 生成序列化JSON文件的路径。默认值： ``''`` ，不指定JSON路径。

    返回：
        Dict，包含序列化数据集图的字典。

    异常：
        - **OSError** - 无法打开文件。
