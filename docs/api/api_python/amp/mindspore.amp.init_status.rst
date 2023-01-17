mindspore.amp.init_status
===========================

.. py:function:: mindspore.amp.init_status()

    初始化溢出状态检测变量。

    .. note::
        该接口仅在Ascend后端有效，在GPU、CPU上调用的返回值没有作用。

    返回：
        Tensor，shape为 (8,) 。
