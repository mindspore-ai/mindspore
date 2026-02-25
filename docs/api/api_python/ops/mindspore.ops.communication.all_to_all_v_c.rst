mindspore.ops.communication.all_to_all_v_c
============================================

.. py:function:: mindspore.ops.communication.all_to_all_v_c(output, input, send_count_matrix, group=None, async_op=False)

    根据用户指定的分割大小，将输入张量分割并发送到其他设备，在接收分割块后合并为单个输出张量。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output** (Tensor) - 从远程rank收集并连接的输出张量。
        - **input** (Tensor) - 要分散到远程rank的张量。
        - **send_count_matrix** (list[int]) - 所有rank的发送和接收参数，:math:`\text{send_count_matrix}[i*\text{rank_size}+j]` 表示rank i发送到rank j的数据量，基本单位是第一维度大小。其中，`rank_size` 表示通信组的大小。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True`` ，CommHandle是一个异步工作句柄。若 `async_op` 是 ``False`` ，CommHandle将返回 ``None`` 。

    异常：
        - **TypeError** - 如果 `input` 或 `output` 不是张量，`group` 不是str，或 `async_op` 不是bool。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
