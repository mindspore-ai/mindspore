mindspore.mint.distributed.destroy_process_group
==================================================

.. py:function:: mindspore.mint.distributed.destroy_process_group(group=None)

    销毁指定通讯group。
    如果指定的通讯group为None或“hccl_world_group”, 则销毁全局通讯域并释放分布式资源，例如 `hccl`  服务。

    .. note::
        - 当前接口不支持GPU、CPU版本的MindSpore调用。
        - 此方法应该在 :func:`mindspore.mint.distributed.init_process_group` 方法之后使用。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **group** (str，可选) - 通信组名称，通常由 :func:`mindspore.mint.distributed.new_group` 方法创建，如果为 ``None`` ， Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。

    异常：
        - **TypeError** - 在参数 `group` 不是字符串时抛出。
        - **RuntimeError** - `HCCL` 服务不可用时，或者使用了MindSpore的GPU或CPU版本。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
