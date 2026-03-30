mindspore.ops.renorm
====================

.. py:function:: mindspore.ops.renorm(input, p, axis, maxnorm)

    返回一个tensor，其中按指定轴计算每个子tensor的 `p` 范数小于等于 `maxnorm`。如果大于，返回子tensor上的原始值除以子tensor的 `p` 范数，然后再乘以 `maxnorm` 。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将发生非兼容性变更：

        - 参数 `axis` 将重命名为 `dim` 。

        变更后的接口签名为 ``mindspore.ops.renorm(input, p, dim, maxnorm)`` 。

    参数：
        - **input** (Tensor) - 输入tensor。
        - **p** (int) - 范数计算的幂。
        - **axis** (int) - 指定计算轴。
        - **maxnorm** (float32) - 指定的最大范数。

    返回：
        Tensor
