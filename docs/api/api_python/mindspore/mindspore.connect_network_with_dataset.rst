mindspore.connect_network_with_dataset
=======================================

.. py:function:: mindspore.connect_network_with_dataset(network, dataset_helper)

    将 `network` 与 `dataset_helper` 中的数据集连接，只支持 `下沉模式 <https://mindspore.cn/tutorials/experts/zh-CN/master/optimize/execution_opt.html>`_，(dataset_sink_mode=True)。

    参数：
        - **network** (Cell) - 数据集的训练网络。
        - **dataset_helper** (DatasetHelper) - 一个处理MindData数据集的类，提供了数据集的类型、形状（shape）和队列名称。

    返回：
        Cell，一个新网络，包含数据集的类型、形状（shape）和队列名称信息。

    异常：
        - **RuntimeError** - 如果该接口在非数据下沉模式调用。
