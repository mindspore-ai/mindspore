mindspore.mint.distributed.all_to_all_single
===============================================

.. py:function:: mindspore.mint.distributed.all_to_all_single(output,input,output_split_sizes=None,input_split_sizes=None,group=None,async_op=False)

    在通信组内的所有进程（rank）之间交换数据。该接口首先根据 `input_split_sizes` 将 `input` 沿第0维切分，
    然后按顺序将各切分块分别发送给通信组内对应的rank。每个rank同时从组内各rank接收数据，
    并根据 `output_split_sizes` 将接收到的数据沿第0维拼接成一个输出张量。

    .. note::
        当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **output** (Tensor) - 表示从远端收集的tensor结果。
        - **input** (Tensor) - 要发送到远端设备的tensor。
        - **output_split_sizes** (Union(Tuple(int), List(int)), 可选) - 接收tensor在0维的切分大小列表。默认值： ``None`` ，表示均匀切分。
        - **input_split_sizes** (Union(Tuple(int), List(int)), 可选) - 发送tensor在0维的切分大小列表。默认值： ``None`` ，表示均匀切分。
        - **group** (str, 可选) - 通信组名称。默认值：``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True``，则CommHandle是一个异步工作句柄；若 `async_op` 是 ``False``，则CommHandle将返回 ``None``。

    异常：
        - **TypeError** - `input` 或者 `output` 不是Tensor类型， `group` 不是str， `async_op` 不是bool。
        - **ValueError** - 当 `input_split_sizes` 为空时， `input` 的第0维不能被通信组内卡数整除。
        - **ValueError** - 当 `output_split_sizes` 为空时， `output` 的第0维不能被通信组内卡数整除。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
