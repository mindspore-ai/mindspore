mindspore.dataset.deserialize
==============================

.. py:method:: mindspore.dataset.deserialize(input_dict=None, json_filepath=None)

    数据处理管道反序列化，支持输入Python字典或使用 `mindspore.dataset.serialize()` 接口生成的JSON文件。

    .. note::
        反序列化包含自定义Python函数的数据处理管道时，部分参数信息可能丢失；当 `input_dict` 和 `json_filepath` 同时不为None时，返回反序列化JSON文件的结果。

    **参数：**

    - **input_dict** (dict) - 以Python字典存储的数据处理管道。默认值：None。
    - **json_filepath** (str) - 数据处理管道JSON文件的路径，该文件以通用JSON格式存储了数据处理管道信息，用户可通过 `mindspore.dataset.serialize()` 接口生成。默认值：None。

    **返回：**

    当反序列化成功时，将返回Dataset对象；当无法被反序列化时，deserialize将会失败，且返回None。

    **异常：**

    - **OSError:** -  `json_filepath` 不为None且JSON文件解析失败时。
