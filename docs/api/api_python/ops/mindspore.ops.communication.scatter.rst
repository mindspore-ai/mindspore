mindspore.ops.communication.scatter
=====================================

.. py:function:: mindspore.ops.communication.scatter(tensor, scatter_list, src=0, group=None, async_op=False)

    在指定通信组中的进程之间均匀分散张量。

    .. note::
        - 接口行为仅支持Tensor列表输入和均匀分散。
        - 只有进程 `src`（全局rank）会执行分散操作。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输出张量。
        - **scatter_list** (list[Tensor]) - 要分散的相同大小的张量列表。默认为None，必须在源rank上指定。
        - **src** (int, 可选) - 指定发送张量的进程的rank（全局rank）。只有进程 `src` 会发送张量。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 如果 `tensor` 参数的类型不是Tensor，`scatter_list` 不是Tensor列表。
        - **TypeError** - 如果 `op` 和 `group` 不是str。async_op不是bool或'op'无效。
        - **TypeError** - 如果 `scatter_list` 的大小不等于组大小。
        - **TypeError** - 如果 `tensor` 的类型或shape不等于 `scatter_list` 的成员。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
