mindspore.ops.communication.gather_into_tensor
================================================

.. py:function:: mindspore.ops.communication.gather_into_tensor(output_tensor, input_tensor, dst=0, group=None, async_op=False)

    从指定通信组中收集张量。操作将根据维度0从进程收集张量。

    .. note::
        - 只有进程 `dst`（全局rank）会保留收集的张量。其他进程将保留一个shape为[1]的张量，没有数学意义。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output_tensor** (Tensor) - 用于容纳来自所有rank的张量元素的输出张量。
        - **input_tensor** (Tensor) - 要收集的张量。张量的shape为 :math:`(x_1, x_2, ..., x_R)`。此API中的输入张量在所有rank之间必须具有相同的大小。
        - **dst** (int, 可选) - 指定接收张量的进程的rank（全局rank）。只有进程 `dst` 会接收收集的张量。默认值： ``0``。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 如果 `input_tensor` 或 `output_tensor` 参数的类型不是Tensor，或 `op` 和 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
