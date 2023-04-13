mindspore.ops.MatrixExp
=======================

.. py:class:: mindspore.ops.MatrixExp

    计算方阵的矩阵指数。支持输入包含batch维。

    更多参考详见 :func:`mindspore.ops.matrix_exp`。

    输入：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(*, n, n)` ，其中 `*` 表示0或更多的batch维。
          支持数据类型：float16、float32、float64、complex64、complex128。

    输出：
        Tensor，其shape和数据类型均与 `x` 相同。
