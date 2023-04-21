mindspore.ops.Cholesky
=======================

.. py:class:: mindspore.ops.Cholesky(upper=False)

    计算单个或成批对称正定矩阵的Cholesky分解。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    更多参考详见 :func:`mindspore.ops.cholesky` 。

    参数：
        - **upper** (bool，可选) - 为 ``True`` 返回上三角矩阵，否则返回下三角矩阵。默认值： ``False`` 。

    输入：
        - **input_x** (Tensor) - shape大小为 :math:`(*, N, N)` ，其中 :math:`*` 是零个或多个由对称正定矩阵组成的批处理维，数据类型为float32或float64。

    输出：
        Tensor，shape和数据类型与 `input_x` 相同。
