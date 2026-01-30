mindspore.ops.clip_by_global_norm
==================================

.. py:function:: mindspore.ops.clip_by_global_norm(x, clip_norm=1.0, use_norm=None)

    通过权重梯度总和的比率来裁剪多个tensor的值。

    .. note::
        - 在半自动并行模式或自动并行模式下，如果输入是梯度，那么将会自动汇聚所有设备上的梯度的平方和。

    参数：
        - **x** (Union(tuple[Tensor], list[Tensor])) - 要被裁剪的数据。
        - **clip_norm** (Union(float, int)) - 裁剪比率，应大于0。默认 ``1.0`` 。
        - **use_norm** (None) - 全局范数。目前只支持None，默认 ``None`` 。

    返回：
        多个tensor组成的tuple。