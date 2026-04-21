mindspore.mint.distributed.all_reduce
=====================================

.. py:function:: mindspore.mint.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=None, async_op=False)

    使用指定方式对通信组内的所有设备的tensor数据进行归约操作，所有设备都得到相同的结果，返回归约操作后的张量。

    .. note::
        - 集合中的所有进程的tensor必须具有相同的shape和格式。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor** (Tensor) - 输入和输出待归约操作的tensor，shape为 :math:`(x_1, x_2, ..., x_R)`，该函数输出直接覆盖输入。
        - **op** (str，可选) - 归约的具体操作，如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、 ``"min"`` 和 ``"avg"`` 。使用 ``"avg"`` 时，结果为求和后除以组内设备数。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 通信组名称。默认值：``None``，即Ascend平台表示使用 ``"hccl_world_group"``。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True``，则CommHandle是一个异步工作句柄；若 `async_op` 是 ``False``，则CommHandle将返回 ``None``。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str， `async_op` 不是bool或者 `op` 值非法。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
