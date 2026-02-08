mindspore.mint.distributed.all_gather
=====================================

.. py:function:: mindspore.mint.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)

    汇聚指定的通信组中的Tensor，并返回汇聚后的Tensor列表。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **tensor_list** (list[Tensor]) - 输出汇聚的Tensor列表。
        - **tensor** (Tensor) - 输入待汇聚操作的Tensor。
        - **group** (str，可选) - 通信组名称。如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是True，则CommHandle是一个异步工作句柄；若 `async_op` 是False，则CommHandle将返回None。

    异常：
        - **TypeError** - 输入 `tensor` 的数据类型不为Tensor， `tensor_list` 不为Tensor列表，`group` 不是str， `async_op` 不是bool。
        - **TypeError** - `tensor_list` 的大小不为通信组大小。
        - **TypeError** -  `tensor` 的数据类型和shape与 `tensor_list` 中所有元素存在不一致。
        - **RuntimeError** - 如果目标设备无效，或后端无效，或分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
 