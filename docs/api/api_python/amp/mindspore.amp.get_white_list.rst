mindspore.amp.get_white_list
==================================

.. py:function:: mindspore.amp.get_white_list()

    提供 `amp_level` 为 ``O1`` 等级的自动混合精度内置白名单拷贝。

    当前的内置白名单内容为：

    [:class:`mindspore.nn.Conv1d`, :class:`mindspore.nn.Conv2d`, :class:`mindspore.nn.Conv3d`,
    :class:`mindspore.nn.Conv1dTranspose`, :class:`mindspore.nn.Conv2dTranspose`,
    :class:`mindspore.nn.Conv3dTranspose`, :class:`mindspore.nn.Dense`, :class:`mindspore.nn.LSTMCell`,
    :class:`mindspore.nn.RNNCell`, :class:`mindspore.nn.GRUCell`, :class:`mindspore.ops.Conv2D`,
    :class:`mindspore.ops.Conv3D`, :class:`mindspore.ops.Conv2DTranspose`,
    :class:`mindspore.ops.Conv3DTranspose`, :class:`mindspore.ops.MatMul`, :class:`mindspore.ops.BatchMatMul`,
    :class:`mindspore.ops.PReLU`, :class:`mindspore.ops.ReLU`, :class:`mindspore.ops.Ger`]

    返回：
        list，内置白名单的拷贝。
