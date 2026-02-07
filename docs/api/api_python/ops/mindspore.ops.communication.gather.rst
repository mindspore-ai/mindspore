mindspore.ops.communication.gather
====================================

.. py:function:: mindspore.ops.communication.gather(tensor, gather_list, dst=0, group=None, async_op=False)

    从指定通信组中收集张量。操作将根据维度0从进程收集张量。

    .. note::
        - 只有进程 `dst` （全局rank）会保留收集的张量。其他进程将保留一个没有数学意义的张量列表。
        - 集合中所有进程的tensor必须具有相同的shape和格式。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 要收集的张量。
        - **gather_list** (list[Tensor]) - 用于收集数据的相同大小的张量列表。
        - **dst** (int, 可选) - 指定接收张量的进程的rank（全局rank）。只有进程 `dst` 会接收收集的张量。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True`` ，CommHandle是一个异步工作句柄。若 `async_op` 是 ``False`` ，CommHandle将返回 ``None`` 。

    异常：
        - **TypeError** - 如果 `tensor` 的类型不是Tensor，或 `gather_list` 不是Tensor列表。
        - **TypeError** - 如果 `dst` 不是整数，`group` 不是字符串或 `async_op` 不是bool。
        - **TypeError** - 如果 `gather_list` 的大小不等于组大小。
        - **TypeError** - 如果 `tensor` 的类型或shape不等于 `gather_list` 的成员。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
