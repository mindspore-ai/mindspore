mindspore.ops.movedim
======================

.. py:function:: mindspore.ops.movedim(x, source, destination)

    调换 `x` 中 `source` 和 `destination` 两个维度的位置。

    其它维度保留在原始位置。

    参数：
        - **x** (Tensor) - 维度需要被移动的的Tensor, `x` 的维度不能是0。
        - **source** (Union[int, sequence[int]]) - 要移动的维度的原始位置。`source` 和 `destination` 长度需一致。
        - **destination** (Union[int, sequence[int]]) - 每个维度的目标位置。`source` 和 `destination` 长度需一致。

    返回：
        维度已经被移动的的Tensor。

    异常：
        - **ValueError** - 如果维度不在 [-x.ndim, x.ndim) 的范围内，或者维度包含重复值。
