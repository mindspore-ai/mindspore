mindspore.communication.comm_func
=================================
集合通信函数式接口。

注意，集合通信函数式接口需要先配置好通信环境变量。

针对Ascend/GPU/CPU设备，推荐使用msrun启动方式，无第三方以及配置文件依赖。详见 `msrun启动 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_ 。

.. py:function:: mindspore.communication.comm_func.all_reduce(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    使用指定方式对通信组内的所有设备的Tensor数据进行规约操作，所有设备都得到相同的结果，返回规约操作后的张量。

    .. note::
        集合中的所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待规约操作的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **op** (str，可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组。默认值：``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。其内容取决于操作。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.all_gather_into_tensor(tensor, group=GlobalComm.WORLD_COMM_GROUP)

    汇聚指定的通信组中的Tensor，并返回汇聚后的张量。

    .. note::
        - 集合中所有进程的Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待汇聚操作的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **group** (str) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，如果组中的device数量为N，则输出的shape为 :math:`(N, x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`group` 不是str。
        - **ValueError** - 调用进程的rank id大于本通信组的rank大小。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.reduce_scatter_tensor(tensor, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约并且分发指定通信组中的张量，返回分发后的张量。

    .. note::
        在集合的所有过程中，Tensor必须具有相同的shape和格式。

    参数：
        - **tensor** (Tensor) - 输入待规约且分发的Tensor，假设其形状为 :math:`(N, *)` ，其中 `*` 为任意数量的额外维度。N必须能够被rank_size整除，rank_size为当前通讯组里面的计算卡数量。
        - **op** (str, 可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，数据类型与 `input_x` 一致，shape为 :math:`(N/rank\_size, *)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 和 `group` 不是字符串。
        - **ValueError** - 如果输入的第一个维度不能被rank size整除。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.reduce(tensor, dst, op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约指定通信组中的张量，并将规约结果发送到目标为dst的进程(全局的进程编号)中，返回发送到目标进程的张量。

    .. note::
        只有目标为dst的进程(全局的进程编号)才会收到规约操作后的输出。
        当前支持pynative模式，不支持graph模式。
        其他进程只得到一个形状为[1]的张量，且该张量没有数学意义。

    参数：
        - **tensor** (Tensor) - 输入待规约的Tensor，Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int) - 指定接收输出的目标进程编号，只有该进程会接收规约操作后的输出结果。
        - **op** (str, 可选) - 规约的具体操作。如 ``"sum"`` 、 ``"prod"`` 、 ``"max"`` 、和 ``"min"`` 。默认值： ``ReduceOp.SUM`` 。
        - **group** (str，可选) - 工作的通信组，默认值： ``GlobalComm.WORLD_COMM_GROUP`` （即Ascend平台为 ``"hccl_world_group"`` ，GPU平台为 ``"nccl_world_group"`` ）。

    返回：
        Tensor，数据类型与输入的 `tensor` 一致，shape为 :math:`(x_1, x_2, ..., x_R)`。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 和 `group` 不是字符串。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在4卡环境下运行。


.. py:function:: mindspore.communication.comm_func.scatter_tensor(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP)

    对输入张量进行均匀散射到通信域的卡上。

    .. note::
        该接口和`pytoch.distributed.scatter`存在行为差异。该接口只支持Tensor输入，且只支持均匀切分。
        只有源为src的进程(全局的进程编号)才会将输入张量作为散射源。
        当前支持pynative模式，不支持graph模式。

    参数：
        - **tensor** (Tensor) - 输入待散射的Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **src** (int，可选) - 表示发送源的进程编号。只有该进程会发送散射源张量。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        Tensor，Tensor第0维等于输入数据第0维除以`src`，其他维度相同
            即 :math:`(x_1/src, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。
    
    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.gather_into_tensor(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP)

    对通信组的输入张量进行聚合。操作会将每张卡的输入Tensor的第0维度上进行聚合，发送到对应卡上。

    .. note::
        只有目标为dst的进程(全局的进程编号)才会收到聚合操作后的输出。其他进程只得到一个形状为[1]的张量，且该张量没有数学意义。
        当前支持pynative模式，不支持graph模式。
        

    参数：
        - **tensor** (Tensor) - 输入待聚合的Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int，可选) - 表示发送源的进程编号。只有该进程会接收聚合后的张量。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        Tensor，Tensor第0维等于输入数据第0维求和，其他shape相同
            即 :math:`(\sum x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - 首个输入的数据类型不为Tensor，`op` 或 `group` 不是str。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.broadcast(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP)

    对输入数据整组广播。

    .. note::
        集合中的所有进程的Tensor的shape和数据格式必须相同。
        当前支持pynative模式，不支持graph模式。

    参数：
        - **tensor** (Tensor) - 输入待广播的Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **src** (int，可选) - 表示发送源的进程编号。只有该进程会广播张量。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    返回：
        Tensor，Tensor的shape与输入相同，即 :math:`(x_1, x_2, ..., x_R)` 。

    异常：
        - **TypeError** - src不是int或group不是str。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.barrier(group=GlobalComm.WORLD_COMM_GROUP)

    同步通信域内的多个进程。进程调用到该算子后进入阻塞状态，直到通信域内所有进程调用到该算子，
    进程被唤醒并继续执行。

    参数：
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    异常：
        - **RuntimeError** - 如果后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.all_to_all_single(tensor, output_split_sizes=None, input_split_sizes=None, group=GlobalComm.WORLD_COMM_GROUP)

    切分张量并散射到通信域内多个进程。

    将输入数据在第0维切分成特定的块数（blocks），并按顺序散射。一般有三个阶段：
    - 准备阶段：入参 `input_split_sizes`, `output_split_sizes`的校验，并计算切分块数(`split_count`)
    - 散射阶段：在每个进程上，操作数沿着第0维拆分为 `split_count` 个块（blocks），且散射到指定的rank上，例如，第i块被发送到第i个rank上。
    - 聚合阶段：每个rank沿着第0维拼接接收到的数据。

    该算子暂不支持不均匀切分并分发，`input_split_sizes` 和 `output_split_sizes` 中元素必须相同。

    .. note::
        聚合阶段，所有进程中的Tensor必须具有相同的shape和格式。
        当前支持pynative模式，不支持graph模式。

    参数：
        - **tensor** (Tensor) - 输入Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **output_split_sizes** (Tuple[int], 可选) - 输出数据的第0维切分大小。该算子暂不支持不均匀切分并分发，`output_split_sizes` 中元素必须相同。默认值：``None``。
        - **input_split_sizes** (Tuple[int], 可选) - 输入数据的第0维切分大小。该算子暂不支持不均匀切分并分发，`output_split_sizes` 中元素必须相同。默认值：``None``。
        - **group** (str, 可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。

    输出：
        Tensor，设输入的shape是 :math:`(x_1, x_2, ..., x_R)`，则输出的shape为 :math:`(y_1, x_2, ..., x_R)`。

    异常：
        - **TypeError** - 如果 `group` 不是字符串。
        - **TypeError** - 如果 `output_split_sizes` 或 `input_split_sizes` 中元素不一致。
        - **TypeError** - 如果 `output_split_sizes` 或 `input_split_sizes` 的求和不等于tensor的第0个维度值。
        - **ValueError** - 如果 `split_count` 无法被 `rank_size` 整除。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.isend(tensor, dst=0, group=GlobalComm.WORLD_COMM_GROUP, tag=0)

    发送张量到指定线程。

    .. note::
        Send 和 Receive 算子需组合使用，且有同一个`sr_tag`。
        当前支持pynative模式，不支持graph模式。

    参数：
        - **tensor** (Tensor) - 输入Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
        - **dst** (int，可选) - 表示发送目标的进程编号。只有目标进程会收到张量。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。
        - **tag** (int，可选) - 用于区分发送、接收消息的标签。该消息将被拥有相同`sr_tag`的Receive接收。

    异常：
        - **TypeError** - dst不是int或group不是str。
        - **ValueError** - 如果该线程的rank id 大于通信组的rank size。

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。


.. py:function:: mindspore.communication.comm_func.irecv(tensor, src=0, group=GlobalComm.WORLD_COMM_GROUP, tag=0)

    发送张量到指定线程。

    .. note::
        Send 和 Receive 算子需组合使用，且有同一个`sr_tag`。
        输入的`tensor`的shape和dtype将用于接收张量，但`tensor`的数据值不起作用。
        当前支持pynative模式，不支持graph模式。

    参数：
        - **tensor** (Tensor) - 输入Tensor。Tensor的shape为 :math:`(x_1, x_2, ..., x_R)` 。
            输入的`tensor`的shape和dtype将用于接收张量，但`tensor`的数据值不起作用。
        - **src** (int，可选) - 表示发送源的进程编号。只会接收来自源进程的张量。
        - **group** (str，可选) - 表示通信域。默认值： ``GlobalComm.WORLD_COMM_GROUP`` 。
        - **tag** (int，可选) - 用于区分发送、接收消息的标签。该消息将被接收来自相同`sr_tag`的Send发送的张量。

    异常：
        - **TypeError** - src不是int或group不是str。
        - **ValueError** - 如果该线程的rank id 大于通信组的rank size。

    .. note::
        .. include:: ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
