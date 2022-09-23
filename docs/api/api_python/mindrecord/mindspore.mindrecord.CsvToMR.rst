
.. py:class:: mindspore.mindrecord.CsvToMR(source, destination, columns_list=None, partition_number=1)

    将CSV格式数据集转换为MindRecord格式数据集。

    .. note::
        示例的详细信息，请参见 `转换CSV数据集 <https://www.mindspore.cn/tutorials/zh-CN/r1.9/advanced/dataset/record.html#转换csv数据集>`_。

    参数：
        - **source** (str) - 待转换的CSV文件路径。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。
        - **columns_list** (list[str]，可选) - CSV中待读取数据列的列表。默认值：None，读取所有的数据列。
        - **partition_number** (int，可选) - 生成MindRecord的文件个数。默认值：1。

    异常：
        - **ValueError** - 参数 `source` 、`destination` 、`partition_number` 无效。
        - **RuntimeError** - 参数 `columns_list` 无效。


    .. py:method:: run()

        执行从CSV格式数据集到MindRecord格式数据集的转换。

        返回：
            MSRStatus，SUCCESS或FAILED。

    .. py:method:: transform()

        :func:`mindspore.mindrecord.CsvToMR.run` 的包装函数来保证异常时正常退出。

        返回：
            SRStatus，SUCCESS或FAILED。
