mindspore.mint.distributed.get_world_size
============================================

.. py:function:: mindspore.mint.distributed.get_world_size(group=None)

    获取指定通信组实例的rank_size。

    .. note::
        - `get_world_size` 方法应该在 :func:`mindspore.mint.distributed.init_process_group` 方法之后使用。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **group** (str，可选) - 通信组名称，通常由 :func:`mindspore.mint.distributed.new_group` 方法创建。默认值： ``None`` ，即在Ascend平台表示为 ``"hccl_world_group"`` 。

    返回：
        int，指定通信组实例的rank_size。如果组不可用，则返回 ``-1`` 。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst

        此示例需要运行在8个设备上。
