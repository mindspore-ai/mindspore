mindspore.ops.clip_by_global_norm
==================================

.. py:function:: mindspore.ops.clip_by_global_norm(x, clip_norm=1.0, use_norm=None)

    通过权重梯度总和的比率来裁剪多个Tensor的值。

    .. note::
        - 输入 `x` 应为Tensor的tuple或list。否则，将引发错误。
        - 在半自动并行模式或自动并行模式下，如果输入是梯度，那么将会自动汇聚所有设备上的梯度的平方和。

    参数：
        - **x** (Union(tuple[Tensor], list[Tensor])) - 由Tensor组成的tuple，其每个元素为任意维度的Tensor。
        - **clip_norm** (Union(float, int)) - 表示裁剪比率，应大于0。默认值：1.0。
        - **use_norm** (None) - 表示全局范数。目前只支持None，默认值：None。

    返回：
        tuple[Tensor]，表示裁剪后的Tensor。其数据类型与 `x` 相同，输出tuple中的每个Tensor与输入shape相同。