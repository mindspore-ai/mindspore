mindspore.ops.get_vm_impl_fn
============================

.. py:function:: mindspore.ops.get_vm_impl_fn(prim)

    通过Primitive对象或Primitive名称，获取虚拟实现函数。

    参数：
        - **prim** (Union[Primitive, str]) - 算子注册的Primitive对象或名称。

    .. note::
        该机制目前适用于调试。

    返回：
        函数，虚拟实现函数。
