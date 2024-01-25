mindspore.scipy.linalg.inv
==========================

.. py:function:: mindspore.scipy.linalg.inv(a, overwrite_a=False, check_finite=True)

    计算矩阵的逆。

    .. note::
        - Windows平台上还不支持 `inv`。
        - 仅支持float32、float64、int32、int64类型的Tensor类型。
        - 如果Tensor是int32、int64类型，它将被强制转换为：mstype.float64类型。

    参数：
        - **a** (Tensor) - 要求逆的方阵。
        - **overwrite_a** (bool, 可选) - 是否覆盖参数 `a` 中的数据（可能会提高性能）。
          默认值：``False``。
        - **check_finite** (bool, 可选) - 是否检查输入矩阵是否只包含有限数。
          禁用可能会带来性能增益，但如果输入确实包含INF或NaN，则可能会导致问题（崩溃、程序不终止）。
          默认值：``True``。

    返回：
        Tensor，矩阵 `a` 的逆。

    异常：
        - **LinAlgError** - 如果 :math:`a` 是单数。
        - **ValueError** - 如果 :math:`a` 不是2D方阵。
