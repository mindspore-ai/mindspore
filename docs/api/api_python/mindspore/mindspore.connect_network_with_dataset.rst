mindspore.connect_network_with_dataset
=======================================

.. py:function:: mindspore.connect_network_with_dataset(network, dataset_helper)

    将 `network` 与 `dataset_helper` 中的数据集连接，只支持下沉模式（dataset_sink_mode=True）。

    参数：
        - **network** (Cell) - 用于训练的神经网络。
        - **dataset_helper** (DatasetHelper) - 处理MindData数据集的类，提供了数据集的类型、形状（shape）和队列名称。

    返回：
        Cell，一个新网络，包含数据集的类型、形状（shape）和队列名称信息。

    异常：
        - **RuntimeError** - 如果该接口在非数据下沉模式下调用。
