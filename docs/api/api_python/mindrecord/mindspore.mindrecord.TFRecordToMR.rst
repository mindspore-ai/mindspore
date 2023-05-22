
.. py:class:: mindspore.mindrecord.TFRecordToMR(source, destination, feature_dict, bytes_fields=None)

    将TFRecord格式数据集转换为MindRecord格式数据集。

    .. note::
        示例的详细信息，请参见 `转换TFRecord数据集 <https://www.mindspore.cn/tutorials/zh-CN/master/advanced/dataset/record.html#转换tfrecord数据集>`_ 。

    参数：
        - **source** (str) - 待转换的TFRecord文件路径。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。
        - **feature_dict** (dict[str, FixedLenFeature]) - TFRecord的feature类别的字典，当前支持
          `FixedLenFeature <https://www.tensorflow.org/api_docs/python/tf/io/FixedLenFeature>`_ 类型。
        - **bytes_fields** (list[str]，可选) - `feature_dict` 中的字节字段，可以为字节类型的图像字段。默认值： ``None`` ，表示没有诸如图像的二进制字段。

    异常：
        - **ValueError** - 无效参数。
        - **Exception** - 找不到TensorFlow模块或其版本不正确。

    .. py:method:: transform()

        执行从TFRecord格式数据集到MindRecord格式数据集的转换。

        .. note::
            请参考类的示例 :class:`mindspore.mindrecord.TFRecordToMR` 。

        返回：
            MSRStatus，SUCCESS或FAILED。

        异常：
            **ParamTypeError** - 设置MindRecord索引字段失败。
            **MRMOpenError** - 新建MindRecord文件失败。
            **MRMValidateDataError** - 原始数据集数据异常。
            **MRMSetHeaderError** - 设置MindRecord文件头失败。
            **MRMWriteDatasetError** - 创建MindRecord索引失败。
            **TypeError** - 参数 `parallel_writer` 不是bool类型。
