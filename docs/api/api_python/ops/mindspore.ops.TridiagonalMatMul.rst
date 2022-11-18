mindspore.ops.TridiagonalMatMul
================================

.. py:class:: mindspore.ops.TridiagonalMatMul

    返回两个矩阵的乘积，其中左边的矩阵是三对角矩阵。

    输入：
        - **superdiag** (Tensor) - 矩阵乘法左边矩阵的高对角线。
          数据类型为float16、float32、double、complex64和complex128。
          shape为 :math:`(..., 1, M)` 。
          最后一个元素被忽略。
        - **maindiag** (Tensor) - 矩阵乘法左边矩阵的主对角线。
          数据类型为float16、float32、double、complex64和complex128。
          shape为 :math:`(..., 1, M)` 。
        - **subdiag** (Tensor) - 矩阵乘法左边矩阵的低对角线。
          数据类型为float16、float32、double、complex64和complex128。
          shape为 :math:`(..., 1, M)` 。
          第一个元素被忽略。
        - **rhs** (Tensor) - 矩阵乘法右边的MxN矩阵。
          数据类型为float16、float32、double、complex64和complex128。
          shape为 :math:`(..., M, N)` 。

    输出：
        Tensor，其数据类型和shape与 `rhs` 一致。

    异常：
        - **TypeError** - 如果 `superdiag` 、 `maindiag` 、 `subdiag` 和 `rhs` 的数据类型不是float16、float32、double、complex64或complex128。
        - **ValueError** - 如果 `superdiag` 、 `maindiag` 、 `subdiag` 的列数不等于 `rhs` 的行数。
        - **ValueError** - 如果 `superdiag` 、 `maindiag` 、 `subdiag` 的行数不等于1。
        - **ValueError** - 如果 `superdiag` 、 `maindiag` 、 `subdiag` 的秩以及 `rhs` 行秩小于2。
        - **ValueError** - 如果 `superdiag` 、 `maindiag` 、 `subdiag` 的shape不相同。
        - **ValueError** - 如果 `superdiag` 、 `maindiag` 、 `subdiag` 和 `rhs` 各自忽略掉最后两个元素之后的shape不一致。
