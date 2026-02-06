mindspore.mint.distributed.gather
=====================================

.. py:function:: mindspore.mint.distributed.gather(tensor, gather_list, dst=0, group=None, async_op=False)

    对通信组的输入张量进行聚合。操作会将每张卡的输入Tensor进行聚合，发送到对应卡上。

    .. note::
        - 只有目标为dst的进程（全局的进程编号）才会收到聚合操作后的输出。其他进程需要传入张量但没有数学意义。
        - 当前仅支持集合中所有进程的Tensor必须具有相同的shape和格式。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入待聚合的Tensor。
        - **gather_list** (list[Tensor]) - 聚合后的Tensor列表。
        - **dst** (int，可选) - 表示发送源的进程编号。只有该进程会接收聚合后的张量。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 输入 `tensor` 的数据类型不为Tensor， `gather_list` 不为Tensor列表。
        - **TypeError** - `dst` 不是int， `group` 不是str或  `async_op` 不是bool。
        - **TypeError** - `gather_list` 的大小不为通信组大小。
        - **TypeError** -  `tensor` 的数据类型和shape与 `gather_list` 中所有元素存在不一致。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
