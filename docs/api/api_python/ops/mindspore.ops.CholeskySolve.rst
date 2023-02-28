mindspore.ops.CholeskySolve
===========================

.. py:class:: mindspore.ops.CholeskySolve(upper=False)

    根据 Cholesky 分解因子 `u` ，计算正定矩阵线性方程组的解。结果表示为 `c` 。

    如果 `upper` 是True， `u` 是上三角形矩阵，可以通过下面公式得到 `c` ：

    .. math::
        c = (u^{T}u)^{{-1}}b

    如果 `upper` 是False， `u` 是下三角形矩阵，可以通过下面公式得到 `c` ：

    .. math::
        c = (uu^{T})^{{-1}}b

    参数：
        - **upper** (bool，可选) - 将Cholesky因子视为下三角矩阵或上三角矩阵的标志。若为True，视为上三角矩阵，反之则为下三角。默认值：False。

    输入：
        - **x1** (Tensor) - 表示二维或三维矩阵的Tensor。Shape为： :math:`(*, N, M)` ，数据类型为float32或float64。
        - **x2** (Tensor) - 表示由2D或3D正方形矩阵组成上三角形或下三角形乔列斯基因子的Tensor。Shape为： :math:`(*, N, N)` ，数据类型为float32或float64。
          `x1` 和 `x2` 必须具有相同的类型。

    输出：
        Tensor，具有与 `x1` 相同的shape和数据类型。

    异常：
        - **TypeError** - `upper` 的数据类型不是bool。
        - **TypeError** - 如果 `x1` 和 `x2` 的dtype不是以下之一：float64，float32。
        - **TypeError** - 如果 `x1` 不是Tensor。
        - **TypeError** - 如果 `x2` 不是Tensor。
        - **ValueError** - 如果 `x1` 和 `x2` 具有不同的批大小。
        - **ValueError** - 如果 `x1` 和 `x2` 具有不同的行数。
        - **ValueError** - 如果 `x1` 不是二维或三维矩阵。
        - **ValueError** - 如果 `x2` 不是二维或三维方阵。
