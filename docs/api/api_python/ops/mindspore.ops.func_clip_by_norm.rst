mindspore.ops.clip_by_norm
============================

.. py:function:: mindspore.ops.clip_by_norm(x, max_norm, norm_type=2.0, error_if_nonfinite=False)

    基于范数对输入Tensor进行裁剪。计算时先将所有输入元素的范数连接成向量，再计算该向量的范数。

    .. warning::
        该接口已废弃，2.9.0（不含）之后版本将删除。

    .. note::
        接口适用于梯度裁剪场景，输入仅支持float类型入参。

    参数：
        - **x** (Union[Tensor, list[Tensor], tuple[Tensor]]) - 需要裁剪的输入。
        - **max_norm** (Union[float, int]) - 该组网络参数的范数上限。
        - **norm_type** (Union[float, int]，可选) - 范数类型。默认 ``2.0`` 。
        - **error_if_nonfinite** (bool，可选) - 若为 ``True`` ，当 `x` 中元素的总范数为nan、inf或-inf时抛出异常；若为 ``False`` ，则不抛出异常。默认 ``False`` 。

    返回：
        Tensor、Tensor的列表或元组，表示裁剪后的Tensor。

    异常：
        - **RuntimeError** - 如果 `x` 的总范数为nan、inf或-inf。
