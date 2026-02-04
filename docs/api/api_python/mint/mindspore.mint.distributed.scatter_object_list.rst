mindspore.mint.distributed.scatter_object_list
===============================================

.. py:function:: mindspore.mint.distributed.scatter_object_list(scatter_object_output_list, scatter_object_input_list, src=0, group=None)

    将输入Python对象列表均匀散射到通信域的卡上。

    .. note::
        - 类似 :func:`mindspore.mint.distributed.scatter` 方法，只支持Python对象列表输入。
        - 只有源为src的进程(全局的进程编号)才会将输入Tensor作为散射源。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **scatter_object_output_list** (list[Any]) - 输出散射的Python对象列表。
        - **scatter_object_input_list** (list[Any]) - 输入待散射的Python对象列表。
        - **src** (int，可选) - 表示发送源的进程编号。只有该进程会发送散射源Tensor。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称，如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。

    异常：
        - **TypeError** - `src` 不是int或 `group` 不是str。
        - **TypeError** - `scatter_object_input_list` 的大小不为通信组大小。
        - **RuntimeError** - 如果目标设备无效，或者后端无效，或者分布式初始化失败。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        该样例需要在2卡环境下运行。
