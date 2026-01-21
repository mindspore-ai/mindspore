mindspore.ops.geqrf
===================

.. py:function:: mindspore.ops.geqrf(input)

    将输入tensor进行QR分解 :math:`A = QR` 。

    分解为正交矩阵 `Q` 和上三角矩阵 `R` 的乘积。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入tensor。

    返回：
        两个tensor组成的tuple( `y` , `tau` )。

        - **y** (Tensor) - 隐式存储 `Q` 和 `R` 矩阵。 `Q` （Householder反射向量）存储在对角线下方， `R` 的元素存储在对角线及上方。
        - **tau** (Tensor) - 存储每个Householder变换的缩放因子（Householder反射系数）。
