
.. py:class:: mindspore.mindrecord.MindPage(file_name, num_consumer=4)

    以分页方式读取MindRecord文件的类。

    参数：
        - **file_name** (Union[str, list[str]]) - MindRecord格式的数据集文件或文件列表。
        - **num_consumer** (int，可选) - 加载数据的并发数。默认值：4。不应小于1或大于处理器的核数。

    异常：
        - **ParamValueError** - `file_name` 、`num_consumer` 或 `columns` 无效。
        - **MRMInitSegmentError** - 初始化ShardSegment失败。

    .. py:method:: candidate_fields
        :property:

        返回用于数据分组的候选category字段。

        返回：
            list[str]，候选category 字段。


    .. py:method:: category_field
        :property:

        返回用于数据分组的category字段。

        返回：
            list[str]，category字段。

    .. py:method:: get_category_fields()

        返回用于数据分组的候选category字段。

        返回：
            list[str]，候选category字段。


    .. py:method:: read_at_page_by_id(category_id, page, num_row)

        以分页方式按category ID进行查询。

        参数：
            - **category_id** (int) - category ID，参考 `read_category_info` 函数的返回值。
            - **page** (int) - 分页的索引。
            - **num_row** (int) - 每个分页的行数。

        返回：
            list[dict]，根据category ID查询的数据。

        异常：
            - **ParamValueError** - 参数无效。
            - **MRMFetchDataError** - 无法按category ID获取数据。
            - **MRMUnsupportedSchemaError** - schema无效。

    .. py:method:: read_at_page_by_name(category_name, page, num_row)

        以分页方式按category字段进行查询。

        参数：
            - **category_name** (str) - category字段对应的字符，参考 `read_category_info` 函数的返回值。
            - **page** (int) - 分页的索引。
            - **num_row** (int) - 每个分页的行数。

        返回：
            list[dict]，根据category字段查询的数据。

    .. py:method:: read_category_info()

        当数据按指定的category字段进行分组时，返回category信息。

        返回：
            str，分组信息的描述。

        异常：
            - **MRMReadCategoryInfoError** - 读取category信息失败。

    .. py:method:: set_category_field(category_field)

        设置category字段。

        .. note::
            必须是候选category字段。

        参数：
            - **category_field** (str) - category字段名称。

        返回：
            MSRStatus，SUCCESS或FAILED
