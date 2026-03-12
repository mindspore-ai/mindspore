mindspore.mint.linalg.qr
=========================

.. py:function:: mindspore.mint.linalg.qr(A, mode='reduced')

    对输入矩阵进行正交分解：:math:`A = QR`。

    其中 `A` 为输入tensor，维度至少为2， `A` 可以表示为正交矩阵 `Q` 与上三角矩阵 `R` 的乘积形式。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **A** (Tensor) - 计算的矩阵，`A` 至少是两维的。
        - **mode** (str，可选) - 矩阵分解的模式，可选 ``"reduced"`` 、 ``"complete"`` 、 ``"r"`` 。默认值： ``"reduced"`` 。

          - ``"reduced"``：对于输入 :math:`A(*, m, n)` 输出简化大小的 :math:`Q(*, m, k)` 、 :math:`R(*, k, n)`，其中k为m、n的最小值。
          - ``"complete"``：对于输入 :math:`A(*, m, n)` 输出完整大小的 :math:`Q(*, m, m)` 、 :math:`R(*, m, n)`。
          - ``"r"``：仅计算reduced场景下的 :math:`R(*, k, n)`，其中k为m、n的最小值，返回Q为空tensor。

    返回：
        - **Q** (Tensor) - shape为 :math:`Q(*, m, k)` 或 :math:`(*, m, m)`，与 `A` 具有相同的dtype。
        - **R** (Tensor) - shape为 :math:`R(*, k, n)` 或 :math:`(*, m, n)`，与 `A` 具有相同的dtype。

    异常：
        - **TypeError** - 如果 `A` 不是Tensor。
        - **TypeError** - 如果 `A` 的dtype不是float32。
        - **ValueError** - 如果 `A` 不为空并且它的维度小于2维。
