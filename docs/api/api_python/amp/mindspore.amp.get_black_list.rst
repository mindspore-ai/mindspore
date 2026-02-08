mindspore.amp.get_black_list
==================================

.. py:function:: mindspore.amp.get_black_list()

    提供用于自动混合精度 `amp_level` 为 ``O2`` 等级的内置黑名单的拷贝。

    当前的内置黑名单内容为：

    [:class:`mindspore.nn.BatchNorm1d`, :class:`mindspore.nn.BatchNorm2d`, :class:`mindspore.nn.BatchNorm3d`,
    :class:`mindspore.nn.LayerNorm`]

    返回：
        list，内置黑名单的拷贝。
