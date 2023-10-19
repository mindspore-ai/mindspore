mindspore.nn.LRN
================

.. py:class:: mindspore.nn.LRN(depth_radius=5, bias=1.0, alpha=1.0, beta=0.5, norm_region="ACROSS_CHANNELS")

    局部响应归一化操作LRN(Local Response Normalization)。

    .. warning::
        LRN在Ascend平台已废弃，存在潜在精度问题。建议使用其他归一化方法，如 :class:`mindspore.nn.BatchNorm1d` 、 :class:`mindspore.nn.BatchNorm2d` 、 :class:`mindspore.nn.BatchNorm3d` 代替LRN。

    更多参考详见 :func:`mindspore.ops.lrn`。
