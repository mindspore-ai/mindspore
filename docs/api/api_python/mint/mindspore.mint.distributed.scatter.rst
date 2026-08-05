mindspore.mint.distributed.scatter
=====================================

.. py:function:: mindspore.mint.distributed.scatter(tensor, scatter_list, src=0, group=None, async_op=False)

    将输入tensor均匀散射到通信域的卡上。

    .. note::
        - 该接口支持Tensor List输入，只支持均匀切分。
        - 只有源为 `src` 的进程（全局进程编号）才会将输入tensor作为散射源。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输出散射的tensor。
        - **scatter_list** (list[Tensor]) - 输入待散射的tensor列表。必须在源 rank 上指定。
        - **src** (int，可选) - 表示发送源的进程编号。只有该进程会发送散射源tensor。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即在Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True`` ，CommHandle是一个异步工作句柄；若 `async_op` 是 ``False`` ，CommHandle将返回 ``None`` 。

    异常：
        - **TypeError** - 输入 `tensor` 的数据类型不为Tensor， `scatter_list` 不为Tensor列表。
        - **TypeError** - `group` 不是str，或 `async_op` 不是bool。
        - **TypeError** - `scatter_list` 的大小不为通信组大小。
        - **TypeError** -  `tensor` 的数据类型和shape与 `scatter_list` 中所有元素存在不一致。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
