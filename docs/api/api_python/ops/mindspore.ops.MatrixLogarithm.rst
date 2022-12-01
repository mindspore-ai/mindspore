mindspore.ops.MatrixLogarithm
=============================

.. py:class:: mindspore.ops.MatrixLogarithm

    返回一个或多个方阵的矩阵对数。

    输入：
        - **x** (Tensor) - `x` 是一个Tensor。Tensor的shape为 :math:`[..., M, M]` 。
          其数据类型必须为complex64或complex128。同时，其shape必须为2D到7D。

    输出：
        - **y** (Tensor) - 与输入的shape和数据类型均相同。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `x` 的数据类型不是complex64或complex128。
        - **ValueError** - 如果 `x` 的维度小于2。
        - **ValueError** - 如果内部的两维不相等。
