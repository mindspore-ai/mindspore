mindspore.mint.distributed.new_group
=====================================

.. py:function:: mindspore.mint.distributed.new_group(ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False, group_desc=None)

    创建用户自定义的通信组实例。

    .. note::
        - `new_group` 方法应该在 :func:`mindspore.mint.distributed.init_process_group` 方法之后使用。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **ranks** (list[int], 可选) - 设备编号列表。默认值为 ``None`` ，将创建全局通信组。
        - **timeout** (int, 无效参数) - 当前为预留参数。
        - **backend** (str, 无效参数) - 当前为预留参数。
        - **pg_options** (GroupOptions, 可选) - 额外通信组配置参数。后端会自动选择支持的参数，并在通信组初始化时生效。例如对于 `HCCL` 后端，可以指定 `hccl_config` 来应用特定的通信组初始化配置。默认值为 ``None`` 。

          `GroupOptions` 被定义为一个可以实例化为Python对象的类。

          .. code-block::

            GroupOptions {
                hccl_config(dict)
            }
            
          `hccl_config` 当前支持 "hccl_buffer_size"、"hccl_comm" 或 "hccl_op_expansion_mode"。

          - **hccl_buffer_size** (uint32)：指定HCCL通信缓冲区的大小。
          - **hccl_comm** (int64)：指定已存在的HcclComm指针。如果设置了 "hccl_comm"，则其他HCCL配置参数无效。
          - **hccl_op_expansion_mode** (str | uint32)：指定通信组的HCCL算子展开模式。支持的字符串值为 "DEFAULT"、"HOST"、"AICPU" 和 "AIV"，支持的整数值为 0、1、2 和 3。

        - **use_local_synchronization** (bool, 无效参数) - 当前为预留参数。
        - **group_desc** (str, 无效参数) - 当前为预留参数。

    返回：
        str，生成的通信组名称。如果执行异常则返回 ``""``。

    异常：
        - **TypeError** - 参数 `ranks` 不是列表或有重复设备号。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
