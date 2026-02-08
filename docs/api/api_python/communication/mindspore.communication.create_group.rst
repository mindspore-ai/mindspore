mindspore.communication.create_group
====================================

.. py:function:: mindspore.communication.create_group(group, rank_ids, options=None)

    创建用户自定义的通信组实例。

    .. note::
        - MindSpore的GPU和CPU版本不支持此方法。
        - 列表rank_ids的长度应大于1。
        - 列表rank_ids内不能有重复数据。
        - `create_group` 方法应该在 `init` 方法之后使用。
        - 如果没有使用mpirun启动，PyNative模式下仅支持全局单通信组。

    参数：
        - **group** (str) - 用户自定义的通信组实例名称。
        - **rank_ids** (list) - 设备编号列表。
        - **options** (GroupOptions, 可选) - 额外通信组配置参数。后端会自动选择支持的参数并在通信组初始化时生效。例如对于 `HCCL` 后端，可以指定 `hccl_config` 来应用特定的通信组初始化配置。默认值为 ``None`` 。

          `GroupOptions` 被定义为一个可以实例化为python对象的类。

          .. code-block::

            GroupOptions {
                hccl_config(dict)
            }

          `hccl_config` 当前仅支持 "hccl_buffer_size" 和 "hccl_comm" 两个参数。

          - **hccl_buffer_size** (uint32)：指定HCCL通信缓冲区的大小。
          - **hccl_comm** (int64)：指定已存在的HcclComm指针。如果设置了 "hccl_comm"，则 "hccl_buffer_size" 无效。

    异常：
        - **TypeError** - 参数 `group` 不是字符串，或参数 `rank_ids` 不是列表。
        - **ValueError** - 列表 `rank_ids` 的长度小于1、列表 `rank_ids` 内有重复数据，或后端无效。
        - **RuntimeError** - `HCCL` 服务不可用，或使用了MindSpore的GPU、CPU版本。

    样例：

    .. note::
        .. include:: ../ops/mindspore.ops.comm_note.rst
