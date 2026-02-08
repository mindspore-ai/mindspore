mindspore.communication.comm_func.all_to_all_v_c
======================================================================

.. py:function:: mindspore.communication.comm_func.all_to_all_v_c(output, input, send_count_matrix, group=None, async_op=False)

    根据用户输入的切分大小，把输入tensor切分后，发送到其他的设备上，并从其他设备接收切分块，然后合并到一个输出tensor中。

    .. note::
        仅支持PyNative模式，目前不支持Graph模式。

    参数：
        - **output** (Tensor) - 表示从远端收集的tensor结果。
        - **input** (Tensor) - 要发送到远端设备的tensor。
        - **send_count_matrix** (list[int]) - 所有rank的收发大小列表。 :math:`\text{send_count_matrix}[i*\text{rank_size}+j]` 表示rank i发给rank j的数据量，基本单位是输入的第一个维度尺寸。其中， `rank_size` 表示通信组大小。
        - **group** (str, 可选) - 工作的通信组。默认值：``None``，表示在Ascend上使用 ``hccl_world_group``，在GPU上使用 ``nccl_world_group``。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True``，CommHandle是一个异步工作句柄；若 `async_op` 是 ``False``，CommHandle将返回 ``None``。

    异常：
        - **TypeError** - `input` 或者 `output` 不是Tensor类型， `group` 不是str， `async_op` 不是bool。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
