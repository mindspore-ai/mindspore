mindspore.dataset.Dataset.recv
==============================

.. py:method:: mindspore.dataset.Dataset.recv(src=0, group=None)

    数据集通信接口，接收源 `Dataset` 使用 :class:`mindspore.dataset.Dataset.send` 发送的数据。

    每调用一次，仅接收一条数据。

    .. note::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **src** (Union[int, list[int]], 可选) - 数据源 `Dataset` 对应的Rank ID或ID列表。指定当前进程的Rank ID时，将直接从自身获取一条数据。默认值： ``0`` ，表示从Rank 0接收数据。
        - **group** (str, 可选) - 指定通信组实例（由 :func:`mindspore.mint.distributed.init_process_group` 或者 :func:`mindspore.mint.distributed.new_group` 方法创建）的名称。默认值： ``None`` ，使用 ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        Union[Tensor, list[Tensor]]，接收到的Tensor/Tensor列表。
