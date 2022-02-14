
.. py:class:: mindspore.mindrecord.FileReader(file_name, num_consumer=4, columns=None, operator=None)

    读取MindRecord格式数据集的类。

    .. note::
        - 如果 `file_name` 是文件名的字符串，则会尝试加载同一批转换生成的所有MindRecord文件，如果缺少其中某个MindRecord文件，则会引发异常。
        - 如果 `file_name` 是文件名的列表，则只加载列表中指定的MindRecord文件。

    **参数：**

    - **file_name** (str, list[str]) - MindRecord格式的数据集文件或文件列表。
    - **num_consumer** (int，可选) - 加载数据的并发数。默认值：4。不应小于1或大于处理器的核数。
    - **columns** ((list[str]，可选) - MindRecord中待读取数据列的列表。默认值：None，读取所有的数据列。
    - **operator** (int，可选) - 保留参数。默认值：None。

    **异常：**

    - **ParamValueError** - `file_name` 、`num_consumer` 或 `columns` 无效。


    .. py:method:: close()

        停止数据集加载并且关闭文件句柄。


    .. py:method:: get_next()

        按列名一次返回一个batch的数据。

    **返回：**

    Dict，key与数据列的列名相同的一个batch的数据。

    **异常：**

    - **MRMUnsupportedSchemaError** - schema无效。
