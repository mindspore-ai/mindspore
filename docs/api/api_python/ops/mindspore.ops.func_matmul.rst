mindspore.ops.matmul
=====================

.. py:function:: mindspore.ops.matmul(input, other)

    返回两个输入的矩阵乘积。

    .. note::
        - `input` 和 `other` 的数据类型必须一致，两者都不能是标量，且须支持广播。
        - Ascend平台， `input` 和 `other` 的秩必须在 1 到 6 之间。
        - 执行反向时，`input` 和 `other` 在即时编译模式动态shape场景下不支持空tensor。

    参数：
        - **input** (Tensor) - 第一个输入tensor。
        - **other** (Tensor) - 第二个输入tensor。

    返回：
        Tensor或Scalar
