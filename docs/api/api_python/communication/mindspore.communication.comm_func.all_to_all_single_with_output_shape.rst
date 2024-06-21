mindspore.communication.comm_func.all_to_all_single_with_output_shape
======================================================================

.. py:function:: mindspore.communication.comm_func.all_to_all_single_with_output_shape(output_shape, tensor, output_split_sizes=None, input_split_sizes=None, group=None)

    根据用户输入的切分大小，把输入tensor切分后，发送到其他的设备上，并从其他设备接收切分块，然后合并到一个输出tensor中。

    .. note::
        各个rank之间发送和接收的切分块大小需要互相匹配。
        仅支持PyNative模式，目前不支持Graph模式。

    参数：
        - **output_shape** (Union(Tensor, Tuple(int))) - 表示接收的张量的形状。
        - **tensor** (Tensor) - 要发送到远端设备的张量。
        - **output_split_sizes** (Union(Tuple(int), List(int))) - 接收张量在0维的切分大小列表。默认值：None，表示均匀切分。
        - **input_split_sizes** (Union(Tuple(int), List(int))) - 发送张量在0维的切分大小列表。默认值：None，表示均匀切分。
        - **group** (str, 可选) - 通信执行所在的通信组。默认值：None。为None时，在Ascend上将使用为 ``hccl_world_group``，在GPU上使用 ``nccl_world_group``。

    返回：
        从远端设备接收分块并合并的张量。如果从其他设备接收的张量为空，它将返回一个没有实际意义的值为0的张量。

    异常:
        - **TypeError** - `tensor` 不是张量类型。
        - **TypeError** - `output_shape` 不是元组或者张量类型。

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在2卡环境下运行。
