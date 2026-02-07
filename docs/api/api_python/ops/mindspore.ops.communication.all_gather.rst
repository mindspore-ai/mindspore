mindspore.ops.communication.all_gather
========================================

.. py:function:: mindspore.ops.communication.all_gather(tensor_list, tensor, group=None, async_op=False)

    从指定通信组中收集张量，并返回收集的张量列表。

    参数：
        - **tensor_list** (list[Tensor]) - 输出列表。
        - **tensor** (Tensor) - 要收集到张量中的输入张量。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True``，CommHandle是一个异步工作句柄；若 `async_op` 是 ``False``，CommHandle将返回 ``None``。

    异常：
        - **TypeError** - 如果输入 `tensor` 的类型不是Tensor，`tensor_list` 不是Tensor列表，`group` 不是str或 `async_op` 不是bool。
        - **TypeError** - 如果 `tensor_list` 的大小不等于组大小。
        - **TypeError** - 如果 `tensor` 的类型或shape不等于 `tensor_list` 的成员。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
