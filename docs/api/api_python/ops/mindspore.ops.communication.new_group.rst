mindspore.ops.communication.new_group
=======================================

.. py:function:: mindspore.ops.communication.new_group(ranks=None, timeout=None, backend=None, pg_options=None, use_local_synchronization=False, group_desc=None)

    创建一个新的分布式组。

    .. note::
        此方法应在 :func:`mindspore.ops.communication.init_process_group` 之后使用。

    参数：
        - **ranks** (list[int], 可选) - 组成员rank的列表。如果为 ``None``，将创建world组。默认值： ``None``。
        - **timeout** (int, 无效) - 当前为预留参数。
        - **backend** (str, 无效) - 支持的后端库，目前支持 ``"hccl"`` 和 ``"mccl"``。当backend为 ``"hccl"`` 时将使用华为集合通信库(HCCL)。当backend为 ``"mccl"`` 时将使用MindSpore集合通信库(MCCL)。如果为 ``None``，在Ascend中表示 ``"hccl"``。默认值： ``None``。
        - **pg_options** (GroupOptions, 可选) - 附加的通信组配置参数。后端将自动选择支持的参数并在组初始化期间应用它们。例如，对于 ``HCCL`` 后端，可以指定 ``hccl_config`，以便可以应用组初始化配置。默认值： ``None``。

            `GroupOptions` 被定义为一个可以作为Python对象实例化的类。

            .. code-block::

                GroupOptions {
                    hccl_config(dict)
                }

            `hccl_config` 目前仅支持 "hccl_buffer_size" 或 "hccl_comm"。

            - hccl_buffer_size (uint32): 指定HCCL通信缓冲区的大小。
            - hccl_comm (int64): 指定现有的HcclComm指针。如果设置了 "hccl_comm"，将忽略 "hccl_buffer_size"。

        - **use_local_synchronization** (bool, 无效) - 当前为预留参数。
        - **group_desc** (str, 无效) - 当前为预留参数。

    返回：
        一个包含组名的字符串。在异常情况下返回 ""。

    异常：
        - **TypeError** - 如果组中的rank列表有重复的rank id。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
