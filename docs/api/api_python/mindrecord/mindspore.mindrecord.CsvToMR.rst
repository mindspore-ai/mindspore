
.. py:class:: mindspore.mindrecord.CsvToMR(source, destination, columns_list=None, partition_number=1)

    将CSV格式数据集转换为MindRecord格式数据集的类。

    .. note::
        示例的详细信息，请参见 `转换CSV数据集 <https://mindspore.cn/docs/programming_guide/zh-CN/master/dataset_conversion.html#转换CSV数据集>`_。

    **参数：**

    - **source** (str) - 待转换的CSV文件路径。
    - **destination** (str) - 转换生成的MindRecord文件路径。
    - **columns_list** (list[str]，可选) - CSV中待读取数据列的列表。默认值：None，读取所有的数据列。
    - **partition_number** (int，可选) - 生成MindRecord的文件个数。默认值：1。

    **异常：**

    - **ValueError** - `source` 、`destination` 、`partition_number` 无效。
    - **RuntimeError** - `columns_list` 无效。


    .. py:method:: run()

        执行从CSV格式数据集到MindRecord格式数据集的转换。

        **返回：**

        MSRStatus，CSV数据集是否成功转换为MindRecord格式数据集。


    .. py:method:: transform()

        :func: `mindspore.mindrecord.CsvToMR.run` 函数的包装函数来保证异常时正常退出。
