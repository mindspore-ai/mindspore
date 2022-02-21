
.. py:class:: mindspore.mindrecord.ImageNetToMR(map_file, image_dir, destination, partition_number=1)

    将ImageNet数据集转换为MindRecord格式数据集的类。

    .. note::
        示例的详细信息，请参见 `Converting the ImageNet Dataset <https:// www.mindspore.cn/docs/programming_guide/en/master/dataset_conversion.html#converting-the-imagenet-dataset>`_。

    **参数：**

    - **map_file** (str) - 标签映射文件的路径。映射文件内容如下：

    .. code-block::

        n02119789 0
        n02100735 1
        n02110185 2
        n02096294 3

    - **image_dir** (str) - ImageNet数据集的目录路径，目录中包含类似n02119789、n02100735、n02110185和n02096294的子目录。
    - **destination** (str) - 转换生成的MindRecord文件路径
    - **partition_number** (int，可选) - 生成MindRecord的文件个数。默认值：1。

    **异常：**

    - **ValueError** - `map_file` 、`image_dir` 或 `destination` 无效。

    .. py:method:: run()

        执行从ImageNet数据集到MindRecord格式数据集的转换。

        **返回：**

        MSRStatus，ImageNet数据集是否成功转换为MindRecord格式数据集。


    .. py:method:: transform()

        :func: `mindspore.mindrecord.ImageNetToMR.run` 函数的包装函数来保证异常时正常退出。
