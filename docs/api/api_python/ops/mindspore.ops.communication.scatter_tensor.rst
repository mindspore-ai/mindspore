mindspore.ops.communication.scatter_tensor
============================================

.. py:function:: mindspore.ops.communication.scatter_tensor(output_tensor, input_tensor, src=0, group=None, async_op=False)

    在指定通信组中的进程之间均匀分散张量。

    .. note::
        - 接口行为仅支持Tensor输入和均匀分散，这与 `pytorch.distributed.scatter` 不同。
        - 只有进程 `src` （全局rank）会执行分散操作。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output_tensor** (Tensor) - 输出张量。它在所有rank之间应具有相同的大小。
        - **input_tensor** (Tensor) - 要分散的输入张量。张量的shape为 :math:`(x_1, x_2, ..., x_R)`。
        - **src** (int, 可选) - 指定发送张量的进程的rank（全局rank）。只有进程 `src` 会发送张量。默认值： ``0``。
        - **group** (str，可选) - 通信组名称。默认值： ``None``  ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True`` ，CommHandle是一个异步工作句柄。若 `async_op` 是 ``False`` ，CommHandle将返回 ``None`` 。

    异常：
        - **TypeError** - 如果 `output_tensor` 或 `input_tensor` 参数的类型不是Tensor，`src` 不是int，`group` 不是str，或 `async_op` 不是bool。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
