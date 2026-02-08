
.. py:class:: mindspore.mindrecord.FileWriter(file_name, shard_num=1, overwrite=False)

    将用户自定义的数据转为MindRecord格式数据集的类。

    .. note::
        生成MindRecord文件后，如果修改文件名，可能会导致读取文件失败。

    参数：
        - **file_name** (str) - 转换生成的MindRecord文件路径。
        - **shard_num** (int，可选) - 生成MindRecord的文件个数。取值范围为[1, 1000]。默认值： ``1`` 。
        - **overwrite** (bool，可选) - 当指定目录存在同名文件时是否覆盖写。默认值： ``False`` 。

    异常：
        - **ParamValueError** - `file_name` 或 `shard_num` 无效。

    .. py:method:: add_index(index_fields)

        指定schema中的字段作为索引来加速MindRecord文件的读取。schema可以通过 `add_schema` 来添加。

        .. note::
            - 索引字段应为Primitive类型，例如 `int` 、`float` 、`str` 。
            - 如果不调用该函数，则默认将schema中所有的Primitive类型的字段设置为索引。
              请参考 :class:`mindspore.mindrecord.FileWriter` 类的样例代码。

        参数：
            - **index_fields** (list[str]) - schema中的字段。

        异常：
            - **ParamTypeError** - 索引字段无效。
            - **MRMDefineIndexError** - 索引字段不是Primitive类型。
            - **MRMAddIndexError** - 无法添加索引字段。
            - **MRMGetMetaError** - 未设置schema或无法获取schema。

    .. py:method:: add_schema(content, desc=None)

        增加描述用户自定义数据的schema。

        .. note::
            请参考 :class:`mindspore.mindrecord.FileWriter` 类的样例代码。

        .. list-table:: MindRecord支持的数据类型
           :widths: 25 25 50
           :header-rows: 1

           * - 数据类型
             - 数据维度
             - 说明
           * - int32
             - /
             - 整数
           * - int64
             - /
             - 整数
           * - float32
             - /
             - 浮点数
           * - float64
             - /
             - 浮点数
           * - string
             - /
             - 字符串
           * - bytes
             - /
             - 二进制
           * - int32
             - [-1] / [-1, 32, 32] / [3, 224, 224]
             - numpy.ndarray类型
           * - int64
             - [-1] / [-1, 32, 32] / [3, 224, 224]
             - numpy.ndarray类型
           * - float32
             - [-1] / [-1, 32, 32] / [3, 224, 224]
             - numpy.ndarray类型
           * - float64
             - [-1] / [-1, 32, 32] / [3, 224, 224]
             - numpy.ndarray类型

        参数：
            - **content** (dict) - schema内容的字典。
            - **desc** (str，可选) - schema的描述。默认值： ``None`` 。

        异常：
            - **MRMInvalidSchemaError** - schema无效。
            - **MRMBuildSchemaError** - 构建schema失败。
            - **MRMAddSchemaError** - 添加schema失败。

    .. py:method:: commit()

        将内存中的数据同步到磁盘，并生成相应的数据库文件。

        .. note::
            请参考 :class:`mindspore.mindrecord.FileWriter` 类的样例代码。

        异常：
            - **MRMOpenError** - 打开MindRecord文件失败。
            - **MRMSetHeaderError** - 设置MindRecord文件的header失败。
            - **MRMIndexGeneratorError** - 创建索引Generator失败。
            - **MRMGenerateIndexError** - 写入数据库失败。
            - **MRMCommitError** - 数据同步到磁盘失败。
            - **RuntimeError** - 并行写失败。

    .. py:method:: open_for_append(file_name)
        :classmethod:

        打开MindRecord文件，准备追加数据。

        参数：
            - **file_name** (str) - MindRecord格式的数据集文件的路径。

        返回：
            FileWriter，MindRecord文件的写对象。

        异常：
            - **ParamValueError** - `file_name` 无效。
            - **FileNameError** - MindRecord文件路径中包含无效字符。
            - **MRMOpenError** - 打开MindRecord文件失败。
            - **MRMOpenForAppendError** - 打开MindRecord文件追加数据失败。

    .. py:method:: set_header_size(header_size)

        设置MindRecord文件的header，其中包含shard信息、schema信息、page的元信息等。
        header越大，MindRecord文件可以存储更多的元信息。如果header大于默认大小（16MB），需要调用本函数来设置合适的大小。

        参数：
            - **header_size** (int) - header大小，单位：字节，可设置范围为16*1024(16KB)到128*1024*1024(128MB)。

        异常：
            - **MRMInvalidHeaderSizeError** - 设置header大小失败。

    .. py:method:: set_page_size(page_size)

        设置存储数据的page大小，page分为两种类型：raw page和blob page。
        page越大，page可以存储更多的数据。如果单个样本大于默认大小（32MB），需要调用本函数来设置合适的大小。

        参数：
            - **page_size** (int) - page大小，单位：字节，可设置范围为32*1024(32KB)到256*1024*1024(256MB)。

        异常：
            - **MRMInvalidPageSizeError** - 设置page大小失败。

    .. py:method:: write_raw_data(raw_data, parallel_writer=False)

        根据schema校验用户自定义数据后，将数据转换为一系列连续的MindRecord格式的数据集文件。

        .. note::
            请参考 :class:`mindspore.mindrecord.FileWriter` 类的样例代码。

        参数：
            - **raw_data** (list[dict]) - 用户自定义数据的列表。
            - **parallel_writer** (bool，可选) - 如果为 ``True`` ，则并行写入用户自定义数据。默认值： ``False`` 。并行写入不支持Windows平台。

        异常：
            - **ParamTypeError** - 索引字段无效。
            - **MRMOpenError** - 打开MindRecord文件失败。
            - **MRMValidateDataError** - 数据校验失败。
            - **MRMSetHeaderError** - 设置MindRecord文件的header失败。
            - **MRMWriteDatasetError** - 写入MindRecord格式的数据集失败。
            - **TypeError** - 参数 `parallel_writer` 不是bool类型。
