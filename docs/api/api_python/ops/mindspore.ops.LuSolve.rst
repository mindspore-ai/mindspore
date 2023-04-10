mindspore.ops.LuSolve
=====================

.. py:class:: mindspore.ops.LuSolve

    使用给定的LU分解A和列向量b，计算线性方程组 :math:`Ay = b` 的解。

    矩阵的LU分解可以由 :func:`mindspore.scipy.linalg.lu` 计算产生。

    .. note::
        `lu_pivots` 的batch维度必须与 `lu_data` 的batch维度相匹配，且batch维度的大小和每个维度的值必须相同。
        例如， `lu_data` 为 :math:`(3, 3, 2, 2)` ， `lu_pivots` 为 :math:`(3, 3, 2)` ， `lu_data`
        的batch维度为 :math:`(3, 3)` ， `lu_pivots` 的batch维度为 :math:`(3, 3)` 。

        `lu_data` 的batch维度必须与 `x` 的batch维度匹配，batch维度可以具有不同的尺寸，从右到左，对应的维度必须相等。
        例如， `lu_data` 为 :math:`(3, 3, 2, 2)` ， `x` 为 :math:`(2, 3, 3, 2, 1)` ，
        `lu_data` 的batch维度为 :math:`(3, 3)` ， `x` 的batch维度为 :math:`(2, 3, 3)` 。

    输入：
        - **x** (Tensor) - 上面等式中的列向量 `b` ，shape为 :math:`(*, m, k)` ，其中 :math:`*` 为batch维度，数据类型float32、float16。
        - **lu_data** (Tensor) - LU分解。shape为 :math:`(*, m, m)` ，其中 :math:`*` 为batch维度。可以分解成一个上三角矩阵U和一个下三角矩阵L，数据类型为float32、float16。
        - **lu_pivots** (Tensor) - LU分解的转置矩阵。shape为 :math:`(*, m)` ，其中 :math:`*` 为batch维度，可以转换成一个置换矩阵P，数据类型int32。

    输出：
        Tensor，数据类型与 `x` 和 `lu_data` 相同。

    异常：
        - **TypeError** - 若 `x` 和 `lu_data` 的数据类型不是float32或float16。
        - **TypeError** - 若 `lu_data` 的数据类型不是int32。
        - **TypeError** - 若 `x` 、 `lu_data` 或 `lu_pivots` 不是Tensor。
        - **TypeError** - 若 `x` 、 `lu_data` 的数据类型不同。
        - **ValueError** - 若 `lu_pivots` 的batch维度与 `lu_data` 的batch维度不匹配。
        - **ValueError** - 若 `x` 的维度小于2， `lu_data` 的维度小于2或 `lu_pivots` 的维度小于1。
