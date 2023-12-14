
.. py:class:: mindspore.mindrecord.Cifar100ToMR(source, destination)

    将CIFAR-100数据集（需要是Python版本的，名字类似：cifar-100-python.tar.gz）转换为MindRecord格式数据集。

    参数：
        - **source** (str) - 待转换的CIFAR-100数据集文件所在目录的路径。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。

    异常：
        - **ValueError** - 参数 `source` 或 `destination` 无效。

    .. py:method:: transform(fields=None)

        执行从CIFAR-100数据集到MindRecord格式数据集的转换。

        .. note::
            请参考 :class:`mindspore.mindrecord.Cifar100ToMR` 类的样例代码。

        参数：
            - **fields** (list[str]，可选) - 索引字段的列表，例如['fine_label', 'coarse_label']。默认值： ``None`` 。
              索引字段的设置请参考函数 :func:`mindspore.mindrecord.FileWriter.add_index` 。

        异常：
            - **ParamTypeError** - 设置MindRecord索引字段失败。
            - **MRMOpenError** - 新建MindRecord文件失败。
            - **MRMValidateDataError** - 原始数据集数据异常。
            - **MRMSetHeaderError** - 设置MindRecord文件头失败。
            - **MRMWriteDatasetError** - 创建MindRecord索引失败。
            - **ValueError** - 参数 `fields` 不合法。
