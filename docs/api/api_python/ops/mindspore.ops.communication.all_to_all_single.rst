mindspore.ops.communication.all_to_all_single
===============================================

.. py:function:: mindspore.ops.communication.all_to_all_single(output, input, output_split_sizes=None, input_split_sizes=None, group=None, async_op=False)

    使用分割大小在所有rank之间分散和收集输入，并在单个张量中返回结果。

    .. note::
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output** (Union(Tensor, Tuple(int))) - 如果函数以in-place模式运行，则是从远程rank收集并连接的输出张量。否则，是表示从远程rank收集并连接的张量shape的张量或shape。
        - **input** (Tensor) - 要分散到远程rank的张量。
        - **output_split_sizes** (Union(Tuple(int), List(int)), 可选) - 维度0的输出分割大小。如果设置为None，则按 ``world_size`` 平均分割。默认值： ``None``。
        - **input_split_sizes** (Union(Tuple(int), List(int)), 可选) - 维度0的输入分割大小。如果设置为None，则按 ``world_size`` 平均分割。默认值： ``None``。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        - CommHandle，若函数以in-place模式运行，返回它。若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。
        - Tuple(Tensor, CommHandle)，若函数以非in-place模式运行，返回它。输出张量是从远程rank收集并连接的。如果从远程收集的张量的numel为零，它将返回一个shape为 `()` 的张量，值没有实际意义。若 `async_op` 是True，CommHandle是一个异步工作句柄。若 `async_op` 是False，CommHandle将返回None。

    异常：
        - **TypeError** - 如果 `input` 或 `output` 不是张量。`group` 不是str，或 `async_op` 不是bool。
        - **ValueError** - 当 `input_split_sizes` 为空时，输入维度0不能被 ``world_size`` 整除。
        - **ValueError** - 当 `output_split_sizes` 为空时，输出维度0不能被 ``world_size`` 整除。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
