mindspore.communication.comm_func.all_to_all_with_output_shape
==============================================================

.. py:function:: mindspore.communication.comm_func.all_to_all_with_output_shape(output_shape_list, input_tensor_list, group=None)

    根据用户输入的张量列表，将对应的张量发送到远端设备，并从其他设备接收张量，返回一个接收的张量列表。

    .. note::
        各个设备之间发送和接收的张量形状需要互相匹配。
        仅支持PyNative模式，目前不支持Graph模式。

    参数：
        - **output_shape_list** - (Union[Tuple(Tensor), List(Tensor), Tuple(Tuple(int))]): 包含接收张量形状的列表。
        - **input_tensor_list** - (Union[Tuple(Tensor), List(Tensor)]): 包含发送到其他设备张量的列表。
        - **group** - (str, 可选): 通信所使用的通信组。默认值:None。为None时，在Ascend上将使用 ``hccl_world_group`` ，在GPU使用 ``nccl_world_group`` 。

    返回：
        Tuple(Tensor)，从远端设备接收的张量列表。

    异常:
        - **TypeError** - 'input_tensor_list' 中不全是张量类型。
        - **TypeError** - 'output_shape_list' 中不全是张量或者元组类型。
        - **TypeError** - 'input_tensor_list' 中张量的数据类型不全部一致。

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst

        该样例需要在多卡环境下运行。
