mindspore.mint.distributed.all_to_all_single
===============================================

.. py:function:: mindspore.mint.distributed.all_to_all_single(output,input,output_split_sizes=None,input_split_sizes=None,group=None,async_op=False)

    根据用户输入的切分大小，把输入tensor切分后，发送到其他的设备上，并从其他设备接收切分块，然后合并到一个输出tensor中。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output** (Tensor) - 表示从远端收集的tensor结果。
        - **input** (Tensor) - 要发送到远端设备的tensor。
        - **output_split_sizes** (Union(Tuple(int), List(int)), 可选) - 接收tensor在0维的切分大小列表。默认值： ``None`` ，表示均匀切分。
        - **input_split_sizes** (Union(Tuple(int), List(int)), 可选) - 发送tensor在0维的切分大小列表。默认值： ``None`` ，表示均匀切分。
        - **group** (str, 可选) - 通信组名称。如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle，若 `async_op` 是True，则CommHandle是一个异步工作句柄；若 `async_op` 是False，则CommHandle将返回None。

    异常：
        - **TypeError** - `input` 或者 `output` 不是tensor类型， `group` 不是str， `async_op` 不是bool。
        - **ValueError** - 当 `input_split_sizes` 为空时， `input` 的第0维不能被通信组内卡数整除。
        - **ValueError** - 当 `output_split_sizes` 为空时， `output` 的第0维不能被通信组内卡数整除。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
