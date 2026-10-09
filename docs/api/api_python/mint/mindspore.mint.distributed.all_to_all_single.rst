mindspore.mint.distributed.all_to_all_single
===============================================

.. py:function:: mindspore.mint.distributed.all_to_all_single(output,input,output_split_sizes=None,input_split_sizes=None,group=None,async_op=False)

    沿第0维切分输入tensor，将第 ``i`` 个切分块发送给通信组内的rank ``i``，并按来源rank顺序将接收的数据原地写入 `output`。
    切分大小表示第0维的长度，而非元素个数。每对rank之间的发送大小与接收大小必须匹配。支持复数tensor，通信后保持原始dtype。

    .. note::
        - 当前仅支持PyNative模式，不支持Graph模式。
        - 输入和输出必须是至少一维的连续tensor，且dtype相同。支持具有非零存储偏移的连续view，输出保留原始存储、shape和dtype。
        - 显式切分列表或元组必须包含与通信组大小相同数量的非负整数，其总和必须等于对应tensor的第0维长度。支持大小为0的切分块。
        - ``None``、空列表或空元组表示对对应tensor进行均匀切分，其第0维必须能被通信组大小整除。
          两侧均省略切分参数或传入空序列时，输入和输出的总元素数必须相同。
        - 异步调用后，应在使用输出的流上调用返回句柄的 ``wait()``。同步调用在内部建立该流依赖，不表示返回时所有设备任务均已在主机侧完成。
        - `group` 使用MindSpore通信组名称，异步返回MindSpore的CommHandle。该操作不支持自动反向传播。

    参数：
        - **output** (Tensor) - 预先分配的输出tensor，用于原地接收从远端收集的结果。
        - **input** (Tensor) - 要发送到远端设备的tensor。
        - **output_split_sizes** (Union(Tuple(int), List(int)), 可选) - 接收tensor在0维的切分大小列表。默认值： ``None`` ，表示均匀切分。
        - **input_split_sizes** (Union(Tuple(int), List(int)), 可选) - 发送tensor在0维的切分大小列表。默认值： ``None`` ，表示均匀切分。
        - **group** (str, 可选) - 通信组名称。默认值：``None`` ，即Ascend平台表示为 ``"hccl_world_group"`` 。
        - **async_op** (bool, 可选) - 本算子是否是异步算子。默认值： ``False`` 。

    返回：
        CommHandle。若 `async_op` 是 ``True``，则CommHandle是一个异步工作句柄；若 `async_op` 是 ``False``，则CommHandle将返回 ``None``。

    异常：
        - **TypeError** - `input` 或者 `output` 不是Tensor类型， `group` 不是str， `async_op` 不是bool，或者切分参数不是由整数构成的列表或元组。
        - **ValueError** - 切分大小为负数、数量或总和不正确，或者均匀切分时第0维不能被通信组大小整除。
        - **ValueError** - 输入或输出为标量、存储不连续、dtype不相同，或者双侧均匀切分时总元素数不相同。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
