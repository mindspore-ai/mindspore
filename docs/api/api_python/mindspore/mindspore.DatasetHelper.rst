mindspore.DatasetHelper
========================

.. py:class:: mindspore.DatasetHelper(dataset, dataset_sink_mode=True, sink_size=-1, epoch_num=1)

    DatasetHelper是处理MindData数据集的类，提供数据集信息。

    根据不同的上下文，改变数据集的迭代方式。可在不同的上下文中使用相同的迭代。

    .. note::
        DatasetHelper的迭代将提供一个epoch的数据。

    参数：
        - **dataset** (Dataset) - 训练数据集迭代器。数据集可以由数据集生成器API在 `mindspore.dataset` 模块中生成，例如 :class:`mindspore.dataset.ImageFolderDataset` 。
        - **dataset_sink_mode** (bool, 可选) - 如果值为 ``True`` ，使用 :class:`mindspore.ops.GetNext` 在设备（Device）上通过数据通道获取数据；否则在主机（Host）直接遍历数据集获取数据。默认值： ``True``。
        - **sink_size** (int, 可选) - 控制每个下沉中的数据量。必须为-1或正数。如果 `sink_size` 为-1，则下沉每个epoch的完整数据集。如果 `sink_size` 大于0，则每个epoch下沉 `sink_size` 条数据。默认值： ``-1``。
        - **epoch_num** (int, 可选) - 控制待发送的epoch数据量。默认值： ``1``。

    .. py:method:: continue_send()

        在epoch开始时，继续向设备发送数据。

    .. py:method:: release()

        释放数据下沉资源。

    .. py:method:: sink_size()

        获取每次迭代的 `sink_size` 。

        返回：
            int，每次迭代的 `sink_size` 值。

    .. py:method:: stop_send()

        停止发送下沉数据。

    .. py:method:: types_shapes()

        从当前配置中的数据集获取类型和形状（shape）。

        返回：
            tuple，包含数据集的类型和形状信息。
