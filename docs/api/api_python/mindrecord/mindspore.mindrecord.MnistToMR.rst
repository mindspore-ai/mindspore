
.. py:class:: mindspore.mindrecord.MnistToMR(source, destination, partition_number=1)

    将MNIST数据集转换为MindRecord格式数据集。

    参数：
        - **source** (str) - 数据集目录路径，其包含t10k-images-idx3-ubyte.gz、train-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz和train-labels-idx1-ubyte.gz数据集文件。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。
        - **partition_number** (int，可选) - 生成MindRecord的文件个数。默认值：1。

    异常：
        - **ValueError** - 参数 `source` 、 `destination` 、 `partition_number` 无效。

    .. py:method:: run()

        执行从MNIST数据集到MindRecord格式数据集的转换。

        返回：
            MSRStatus，MNIST数据集是否成功转换为MindRecord格式数据集。

    .. py:method:: transform()

        :func:`mindspore.mindrecord.MnistToMR.run` 函数的包装函数来保证异常时正常退出。

        返回：
            MSRStatus，SUCCESS或FAILED。
