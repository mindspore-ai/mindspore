mindspore.DatasetHelper
========================

.. py:class:: mindspore.DatasetHelper(dataset, dataset_sink_mode=True, sink_size=-1, epoch_num=1)

    DatasetHelper是一个处理MindData数据集的类，提供数据集信息。

    根据不同的上下文，改变数据集的迭代，在不同的上下文中使用相同的迭代。

    .. note::
        DatasetHelper的迭代将提供一个epoch的数据。

    **参数：**

    - **dataset** (Dataset) - 训练数据集迭代器。数据集可以由数据集生成器API在 :class:`mindspore.dataset` 中生成，例如 :class:`mindspore.dataset.ImageFolderDataset` 。
    - **dataset_sink_mode** (bool) - 如果值为True，使用 :class:`mindspore.ops.GetNext` 在设备（Device）上通过数据通道中获取数据，否则在主机直接遍历数据集获取数据。默认值：True。
    - **sink_size** (int) - 控制每个下沉中的数据量。如果 `sink_size` 为-1，则下沉每个epoch的完整数据集。如果 `sink_size` 大于0，则下沉每个epoch的 `sink_size` 数据。默认值：-1。
    - **epoch_num** (int) - 控制待发送的epoch数据量。默认值：1。

    **样例：**

    >>> from mindspore import DatasetHelper
    >>>
    >>> train_dataset = create_custom_dataset()
    >>> set_helper = DatasetHelper(train_dataset, dataset_sink_mode=False)
    >>> # DatasetHelper对象是可迭代的
    >>> for next_element in set_helper:
    ...     next_element
    
    .. py:method:: continue_send()
        
        在epoch开始时继续向设备发送数据。

    .. py:method:: dynamic_min_max_shapes()
        
        返回动态数据的形状(shape)范围（最小形状(shape)，最大形状(shape)）。

        **样例：**

        >>>from mindspore import DatasetHelper
        >>>
        >>>train_dataset = create_custom_dataset()
        >>># config dynamic shape
        >>>dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": [None]})
        >>>dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>>
        >>>min_shapes, max_shapes = dataset_helper.dynamic_min_max_shapes()


    .. py:method:: get_data_info()
        
        下沉模式下，获取当前批次数据的类型和形状(shape)。通常在数据形状(shape)动态变化的场景使用。
    
        **样例：**

        >>> from mindspore import DatasetHelper
        >>>
        >>> train_dataset = create_custom_dataset()
        >>> dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>>
        >>> types, shapes = dataset_helper.get_data_info()

    .. py:method:: release()
        
        释放数据下沉资源。

    .. py:method:: sink_size()
        
        获取每次迭代的 `sink_size` 。

        **样例：**

        >>>from mindspore import DatasetHelper
        >>>
        >>>train_dataset = create_custom_dataset()
        >>>dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True, sink_size=-1)
        >>>
        >>># if sink_size==-1, then will return the full size of source dataset.
        >>>sink_size = dataset_helper.sink_size()

    .. py:method:: stop_send()
        
        停止发送数据下沉数据。

    .. py:method:: types_shapes()
        
        从当前配置中的数据集获取类型和形状(shape)。

        **样例：**

        >>>from mindspore import DatasetHelper
        >>>
        >>>train_dataset = create_custom_dataset()
        >>>dataset_helper = DatasetHelper(train_dataset, dataset_sink_mode=True)
        >>>
        >>>types, shapes = dataset_helper.types_shapes()