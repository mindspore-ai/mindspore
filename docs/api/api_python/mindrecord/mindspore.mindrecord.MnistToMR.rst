
.. py:class:: mindspore.mindrecord.MnistToMR(source, destination, partition_number=1)

    将MNIST数据集转换为MindRecord格式数据集。

    参数：
        - **source** (str) - 数据集目录路径，其包含t10k-images-idx3-ubyte.gz、train-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz和train-labels-idx1-ubyte.gz数据集文件。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。
        - **partition_number** (int，可选) - 生成MindRecord的文件个数。默认值： ``1`` 。

    异常：
        - **ValueError** - 参数 `source` 、 `destination` 、 `partition_number` 无效。

    .. py:method:: transform()

        执行从MNIST数据集到MindRecord格式数据集的转换。

        .. note::
            请参考 :class:`mindspore.mindrecord.MnistToMR` 类的样例代码。

        异常：
            - **ParamTypeError** - 设置MindRecord索引字段失败。
            - **MRMOpenError** - 新建MindRecord文件失败。
            - **MRMValidateDataError** - 原始数据集数据异常。
            - **MRMSetHeaderError** - 设置MindRecord文件头失败。
            - **MRMWriteDatasetError** - 创建MindRecord索引失败。
