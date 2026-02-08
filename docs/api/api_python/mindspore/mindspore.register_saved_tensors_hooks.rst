mindspore.register_saved_tensors_hooks
======================================

.. py:function:: mindspore.register_saved_tensors_hooks(pack_hook, unpack_hook)

    一个静态图模式下的装饰器，用于自定义保存张量（Saved Tensor）的打包（pack）和解包（unpack）方式。

    功能上等价于动态图模式的 `with mindspore.saved_tensors_hooks(pack_hook, unpack_hook)`。
    更多详细信息请参考 :class:`mindspore.saved_tensors_hooks`。

    .. note::
        - 该装饰器只支持图模式。
        - `pack_hook` 和 `unpack_hook` 必须满足图模式下的语法约束。
