mindspore.mint.distributed.all_to_all
=====================================

.. py:function:: mindspore.mint.distributed.all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)

    根据用户输入的张量列表，将对应的张量发送到远端设备，并从其他设备接收张量，返回一个接收的张量列表。

    .. note::
        - 各个设备之间发送和接收的张量形状需要互相匹配。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output_tensor_list** (List[Tensor]) - 包含接收张量的列表。
        - **input_tensor_list** (List[Tensor]) - 包含发送到其他设备张量的列表。
        - **group** (str, 可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - `input_tensor_list` 和 `output_tensor_list` 中不全是张量类型。
        - **TypeError** - `input_tensor_list` 和 `output_tensor_list` 中张量的数据类型不全部一致。
        - **TypeError** - `group` 不是str， `async_op` 不是bool。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
