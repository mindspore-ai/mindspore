mindspore.dataset.serialize
============================

.. py:function:: mindspore.dataset.serialize(dataset, json_filepath='')

    将数据处理管道序列化成JSON文件。

    .. note::
        目前不支持某些Python对象序列化。对于map算子的自定义Python函数序列化， `mindspore.dataset.serialize` 仅返回其函数名称。

    参数：
        - **dataset** (Dataset) - 数据处理管道对象。
        - **json_filepath** (str) - 生成序列化JSON文件的路径。

    返回：
        Dict，包含序列化数据集图的字典。

    异常：
        - **OSError** - 无法打开文件。
