mindspore.mint.distributed.all_gather_into_tensor
===================================================

.. py:function:: mindspore.mint.distributed.all_gather_into_tensor(output_tensor, input_tensor, group=None, async_op=False)

    汇聚指定的通信组中的tensor，并返回汇聚后的tensor。

    .. note::
        - 集合中所有进程的tensor必须具有相同的shape和格式。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output_tensor** (Tensor) - 输出待汇聚操作的tensor。如果组中的device数量为N，则输出tensor的shape为 :math:`(N*x_1, x_2, ..., x_R)` 。
        - **input_tensor** (Tensor) - 输入待汇聚操作的tensor，shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True``，则CommHandle是一个异步工作句柄；若 `async_op` 是 ``False``，则CommHandle将返回 ``None``。

    异常：
        - **TypeError** - `output_tensor` 或 `input_tensor` 输入的数据类型不为Tensor， `group` 不是str， `async_op` 不是bool。
        - **RuntimeError** - 如果目标设备无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
