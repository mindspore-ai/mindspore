
.. py:class:: mindspore.mindrecord.CsvToMR(source, destination, columns_list=None, partition_number=1)

    将CSV格式数据集转换为MindRecord格式数据集。

    参数：
        - **source** (str) - 待转换的CSV文件路径。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。
        - **columns_list** (list[str]，可选) - CSV中待读取数据列的列表。默认值： ``None`` ，读取所有的数据列。
        - **partition_number** (int，可选) - 生成MindRecord的文件个数。默认值： ``1`` 。

    异常：
        - **ValueError** - 参数 `source` 、`destination` 、`partition_number` 无效。
        - **RuntimeError** - 参数 `columns_list` 无效。

    .. py:method:: transform()

        执行从CSV格式数据集到MindRecord格式数据集的转换。

        .. note::
            请参考 :class:`mindspore.mindrecord.CsvToMR` 类的样例代码。

        返回：
            MSRStatus，SUCCESS或FAILED。

        异常：
            - **ParamTypeError** - 设置MindRecord索引字段失败。
            - **MRMOpenError** - 新建MindRecord文件失败。
            - **MRMValidateDataError** - 原始数据集数据异常。
            - **MRMSetHeaderError** - 设置MindRecord文件头失败。
            - **MRMWriteDatasetError** - 创建MindRecord索引失败。
            - **IOError** - 参数 `source` 不存在。
            - **ValueError** - CSV文件首行为列名，每个字段不能以数字开头。
