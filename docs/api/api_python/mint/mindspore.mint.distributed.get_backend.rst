mindspore.mint.distributed.get_backend
==========================================

.. py:function:: mindspore.mint.distributed.get_backend(group=None)

    在指定通信组中获取当前分布式后端的名称。

    .. note::
        - 后端类型包含 ``"hccl"`` 、 ``"nccl"`` 和 ``"mccl"`` ，当前仅支持 ``"hccl"`` 和 ``"mccl"`` 。
        - 当前仅支持PyNative模式，不支持Graph模式。

    参数：
        - **group** (str，可选) - 通信组名称，通常由 :func:`mindspore.mint.distributed.new_group` 方法创建，如果为 ``None`` ， Ascend平台表示为 ``"hccl_world_group"`` 。 默认值： ``None`` 。

    返回：
        str，分布式后端的名称。

    异常：
        - **TypeError** - 参数 `group` 不是字符串。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
