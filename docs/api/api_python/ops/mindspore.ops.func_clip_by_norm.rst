mindspore.ops.clip_by_norm
============================

.. py:function:: mindspore.ops.clip_by_norm(x, max_norm, norm_type=2.0, error_if_nonfinite=False)

    对一组输入Tensor进行范数裁剪。该范数是将输入中的所有元素分别计算范数后连接成一个向量后再计算范数的结果。

    .. note::
        接口适用于梯度裁剪场景，输入仅支持float类型入参。

    参数：
        - **x** (Union(Tensor, list[Tensor], tuple[Tensor])) - 希望进行裁剪的输入。
        - **max_norm** (Union(float, int)) - 该组网络参数的范数上限。
        - **norm_type** (Union(float, int)) - 范数类型。默认值： ``2.0`` 。
        - **error_if_nonfinite** (bool) - 若为 ``True`` ，当 `x` 中元素的总范数为nan、inf或-inf，抛出异常。 若为 ``False``，则不抛出异常。默认值为 ``False`` 。
    返回：
        Tensor、Tensor的列表或元组，表示裁剪后的Tensor。
    
    异常：
        - **RuntimeError** - 如果 `x` 的总范数为nan、inf或-inf。
