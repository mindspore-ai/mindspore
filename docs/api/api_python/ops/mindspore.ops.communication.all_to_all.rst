mindspore.ops.communication.all_to_all
========================================

.. py:function:: mindspore.ops.communication.all_to_all(output_tensor_list, input_tensor_list, group=None, async_op=False)

    根据输入/输出张量列表，在所有rank之间分散和收集张量列表。

    .. note::
        - `output_tensor_list` 和 `input_tensor_list` 中的张量shape应在rank之间匹配。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output_tensor_list** (Union[List(Tensor), List(Tuple(int))]) - 如果函数以in-place模式运行，则表示从远程rank收集的张量列表。否则，表示从远程rank收集的张量或shape列表。
        - **input_tensor_list** (List[Tensor]) - 要分散到远程rank的张量列表。
        - **group** (str，可选) - 通信组名称。默认值： ``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        - 若函数以in-place模式运行，返回CommHandle。
        - 若函数以非in-place模式运行，返回Tuple(Tensor, CommHandle)。第一个元素存储输出结果，第二个元素是CommHandle。

        其中，当 `async_op` 是 ``True`` ，则CommHandle是一个异步工作句柄；当 `async_op` 是 ``False`` ，则CommHandle将返回 ``None`` 。

    异常：
        - **TypeError** - 如果 `input_tensor_list` 或 `output_tensor_list` 中的元素不全是Tensor。
        - **TypeError** - 如果 `input_tensor_list` 或 `output_tensor_list` 中的张量不是同一类型。
        - **TypeError** - 如果 `group` 不是str或 `async_op` 不是bool。

    样例：

    .. note::
        .. include:: ../mint/mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
