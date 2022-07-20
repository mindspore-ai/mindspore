mindspore.connect_network_with_dataset
=======================================

.. py:function:: mindspore.connect_network_with_dataset(network, dataset_helper)

    将 `network` 与 `dataset_helper` 中的数据集连接。

    此函数使用 :class:`mindspore.ops.GetNext` 包装输入网络，以便在正向计算期间可以自动从与队列名称对应的数据通道中提取数据，并将数据传递到输入网络。

    .. note::
        如果以图模式在Ascend/GPU上运行网络，此函数将使用 :class:`mindspore.ops.GetNext` 包装输入网络。在其他情况下，输入网络将在没有改动的情况下返回。仅在下沉模式下获取数据需要使用 :class:`mindspore.ops.GetNext` ，因此此函数不适用于非下沉模式。

    参数：
        - **network** (Cell) - 数据集的训练网络。
        - **dataset_helper** (DatasetHelper) - 一个处理MindData数据集的类，提供了数据集的类型、形状（shape）和队列名称，以包装 :class:`mindspore.ops.GetNext` 。

    返回：
        Cell，在Ascend上以图模式运行任务的情况下，一个由 :class:`mindspore.ops.GetNext` 包装的新网络。在其他情况下是输入网络。

    异常：
        - **RuntimeError** - 如果该接口在非数据下沉模式调用。
