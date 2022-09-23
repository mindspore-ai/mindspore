
.. py:class:: mindspore.mindrecord.Cifar100ToMR(source, destination)

    将CIFAR-100数据集转换为MindRecord格式数据集。

    .. note::
        示例的详细信息，请参见 `转换CIFAR-10数据集 <https://www.mindspore.cn/tutorials/zh-CN/r1.9/advanced/dataset/record.html#转换cifar-10数据集>`_。

    参数：
        - **source** (str) - 待转换的CIFAR-100数据集文件所在目录的路径。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。

    异常：
        - **ValueError** - 参数 `source` 或 `destination` 无效。

    .. py:method:: run(fields=None)

        执行从CIFAR-100数据集到MindRecord格式数据集的转换。

        参数：
            - **fields** (list[str]，可选) - 索引字段的列表，例如['fine_label', 'coarse_label']。默认值：None。
              索引字段的设置请参考函数 :func:`mindspore.mindrecord.FileWriter.add_index` 。

        返回：
            MSRStatus，SUCCESS或FAILED。

    .. py:method:: transform(fields=None)

        :func:`mindspore.mindrecord.Cifar100ToMR.run` 的包装函数来保证异常时正常退出。

        参数：
            - **fields** (list[str]，可选) - 索引字段的列表，例如['fine_label', 'coarse_label']。默认值：None。
              索引字段的设置请参考函数 :func:`mindspore.mindrecord.FileWriter.add_index` 。

        返回：
            MSRStatus，SUCCESS或FAILED。
