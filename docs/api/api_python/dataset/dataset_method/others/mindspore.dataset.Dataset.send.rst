mindspore.dataset.Dataset.send
==============================

.. py:method:: mindspore.dataset.Dataset.send(tensor=None, dst=0, group=None)

    数据集通信接口，将数据发送至目标 `Dataset` ，可以通过 :class:`mindspore.dataset.Dataset.recv` 接收。

    每调用一次，仅发送一条数据。

    .. note::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **tensor** (Union[Tensor, list[Tensor]], 可选) - 待发送的Tensor/Tensor列表。默认值： ``None`` ，表示从当前数据集中获取数据并发送。
        - **dst** (Union[int, list[int]], 可选) - 目标 `Dataset` 对应的Rank ID或ID列表。不能是当前Rank ID或包含当前Rank ID。默认值： ``0`` ，表示将数据发送到Rank 0。
        - **group** (str, 可选) - 指定通信组实例（由 :func:`mindspore.mint.distributed.init_process_group` 或者 :func:`mindspore.mint.distributed.new_group` 方法创建）的名称。默认值： ``None`` ，使用 ``GlobalComm.WORLD_COMM_GROUP`` 。
