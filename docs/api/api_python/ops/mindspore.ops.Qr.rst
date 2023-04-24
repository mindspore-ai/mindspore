mindspore.ops.Qr
=================

.. py:class:: mindspore.ops.Qr(full_matrices=False)

    返回一个或多个矩阵的QR（正交三角）分解。如果 `full_matrices` 设为 ``True`` ，则计算全尺寸q和r，如果为 ``False`` （默认值），则计算q的P列，其中P是 `x` 的2个最内层维度中的最小值。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **full_matrices** (bool，可选) - 是否进行全尺寸的QR分解。默认值： ``False`` 。

    输入：
        - **x** (Tensor) - 要进行分解的矩阵。矩阵必须至少为二维。数据类型：float16、float32、float64、complex64、complex128。
          将 `x` 的shape定义为 :math:`(..., m, n)` ，p定义为m和n的最小值。

    输出：
        - **q** (Tensor) - `x` 的正交矩阵。如果 `full_matrices` 为True，则shape为 :math:`(m, m)` ，否则shape为 :math:`(m, p)` 。 `q` 的数据类型与 `x` 相同。
        - **r** (Tensor) - `x` 的上三角形矩阵。如果 `full_matrices` 为True，则shape为 :math:`(m, n)` ，否则shape为 :math:`(p, n)` 。 `r` 的数据类型与 `x` 相同。

    异常：
        - **TypeError** - `x` 不是Tensor。
        - **TypeError** - `full_matrices` 不是bool。
        - **ValueError** - `x` 的维度小于2。

