mindspore.ops.sum_to_size
=========================

.. py:function:: mindspore.ops.sum_to_size(x, *size)

    将Tensor `x` 加和成 `size`。`size` 必须可以扩展到Tensor的大小。

    参数：
        - **x** (Tensor) - 求和的Tensor。
        - **size** (Union[tuple(int), int]) - 期望输出Tensor的shape。

    返回：
        Tensor，根据 `size` 对 `x` 进行求和的结果。

    异常：
        - **ValueError** - `size` 不能扩展成 `x` 的大小。
