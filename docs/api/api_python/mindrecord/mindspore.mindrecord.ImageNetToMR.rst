
.. py:class:: mindspore.mindrecord.ImageNetToMR(map_file, image_dir, destination, partition_number=1)

    将ImageNet数据集转换为MindRecord格式数据集。

    参数：
        - **map_file** (str) - 标签映射文件的路径。该文件可通过命令： :code:`ls -l [image_dir] | grep -vE "总用量|total|\." | awk -F " " '{print $9, NR-1;}' > [file_path]` 生成，其中 `image_dir` 为ImageNet数据集的目录路径， `file_path` 为生成的 `map_file` 文件 。 `map_file` 文件内容示例如下：

          .. code-block::

              n01440764 0
              n01443537 1
              n01484850 2
              n01491361 3
              ...
              n15075141 999

        - **image_dir** (str) - ImageNet数据集的目录路径，目录中包含类似n01440764、n01443537、n01484850和n15075141的子目录。
        - **destination** (str) - 转换生成的MindRecord文件路径，需提前创建目录并且目录下不能存在同名文件。
        - **partition_number** (int，可选) - 生成MindRecord的文件个数。默认值：1。

    异常：
        - **ValueError** - 参数 `map_file` 、`image_dir` 或 `destination` 无效。

    .. py:method:: run()

        执行从ImageNet数据集到MindRecord格式数据集的转换。

        返回：
            MSRStatus，SUCCESS或FAILED。

    .. py:method:: transform()

        封装 :func:`mindspore.mindrecord.ImageNetToMR.run` 函数来保证异常时正常退出。

        返回：
            MSRStatus，SUCCESS或FAILED。
