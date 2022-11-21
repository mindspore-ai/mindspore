mindspore.dataset.Schema
=========================

.. py:class:: mindspore.dataset.Schema(schema_file=None)

    用于解析和存储数据列属性的类。

    参数：
        - **schema_file** (str) - schema文件的路径。默认值：None。

    返回：
        schema对象，关于数据集的行列配置的策略信息。

    异常：
        - **RuntimeError** - 模式文件加载失败。

    .. py:method:: add_column(name, de_type, shape=None)

        向schema中添加新列。

        参数：
            - **name** (str) - 列的新名称。
            - **de_type** (str) - 列的数据类型。
            - **shape** (list[int], 可选) - 列shape。默认值：None，-1表示该维度的shape是未知的。

        异常：
            - **ValueError** - 列类型未知。
        
    .. py:method:: from_json(json_obj)

        从JSON对象获取schema文件。

        参数：
            - **json_obj** (dictionary) - 解析的JSON对象。

        异常：
            - **RuntimeError** - 对象中存在未知的项。
            - **RuntimeError** - 对象中缺少数据集类型。
            - **RuntimeError** - 对象中缺少列。

    .. py:method:: parse_columns(columns)

        解析传入的数据列的属性并将其添加到自身的schema中。

        参数：
            - **columns** (Union[dict, list[dict], tuple[dict]]) - 数据集属性信息，从schema文件解码。

              - **list** [dict]：'name'和 'type'必须为key值， 'shape'可选。
              - **dict** ：columns.keys()作为名称，columns.values()是dict，其中包含 'type'， 'shape'可选。

        异常：
            - **RuntimeError** - 解析列失败。
            - **RuntimeError** - 列name字段缺失。
            - **RuntimeError** - 列type字段缺失。

    .. py:method:: to_json()

        获取schema的JSON字符串。

        返回：
            str，模式的JSON字符串。
        