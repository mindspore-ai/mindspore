mindspore.mint.distributed.recv_object_list
===========================================

.. py:function:: mindspore.mint.distributed.recv_object_list(object_list, src=0, group=None, device=None)

    同步接收源的进程的Python对象列表。

    .. note::
        - 当前仅支持PyNative模式，不支持Graph模式。
        - 类似于 :func:`mindspore.mint.distributed.recv`，但可以接收Python对象。

    .. warning::
        本接口会隐式使用pickle模块，该模块是不安全的。核心原因是反序列化时会执行任意代码，攻击者可通过构造恶意对象触发系统命令执行。因此，调用者须自行保障接口使用数据的安全性。

    参数：
        - **object_list** (list[Any]) - 要接收对象的列表。必须提供与发送列表大小相等的列表。
        - **src** (int，可选) - 表示发送源的全局进程编号。默认值： ``0`` 。
        - **group** (str，可选) - 通信组名称。如果为 ``None`` ，Ascend平台表示为 ``"hccl_world_group"`` 。默认值： ``None`` 。
        - **device** (str，可选) - 目前这是一个保留参数。默认值： ``None`` 。

    返回：
        int。若成功，返回 ``0``。

    异常：
        - **TypeError** - 如果 `src` 不是int类型、 `group` 不是str类型，或者 `object_list` 不是list类型。
        - **ValueError** - 如果 `src` 线程和当前线程号相同。

    样例：

    .. note::
        .. include:: mindspore.mint.comm_note.rst
