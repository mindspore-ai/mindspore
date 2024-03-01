mindspore.scipy.linalg.eigh
===========================

.. py:function:: mindspore.scipy.linalg.eigh(a, b=None, lower=True, eigvals_only=False, overwrite_a=False, overwrite_b=False, turbo=True, eigvals=None, type=1, check_finite=True)

    求解复Hermitian矩阵或实对称矩阵的标准或广义特征值问题。

    求出 `a` 的特征值Tensor `w` 和可选的特征值Tensor `v`，其中 `b` 是正定的，使得对于每个特征值 `λ` （ `w` 的第i个条目）及其特征向量 `vi` （ `v` 的第i列）满足：

    .. code-block::

                      a @ vi = λ * b @ vi
        vi.conj().T @ a @ vi = λ
        vi.conj().T @ b @ vi = 1

    在标准问题中，假设 `b` 是单位矩阵。

    .. note::
        - Windows平台上还不支持 `eigh`。
        - 如果Tensor是int32、int64类型，它将被强制转换为：mstype.float64类型。

    参数：
        - **a** (Tensor) - 一个shape 为 :math:`(M,M)` 的复Hermitian矩阵或实对称矩阵，用于计算其特征值和特征向量。
        - **b** (Tensor, 可选) - 一个shape为 :math:`(M,M)` 的复Hermitian矩阵或实对称正矩阵。
          如果缺省，则假定为传入单位矩阵。
          默认值：``None``。
        - **lower** (bool, 可选) - 控制相关的Tensor数据是取自 `a` 和 `b` 的下三角还是上三角。
          默认值：``True``。
        - **eigvals_only** (bool, 可选) - 是否只计算特征值，不计算特征向量。
          默认值：``False``。
        - **overwrite_a** (bool, 可选) - 是否覆盖 `a` 中的数据（可能会提高性能）。
          默认值：``False``。
        - **overwrite_b** (bool, 可选) - 是否覆盖 `b` 中的数据（可能会提高性能）。
          默认值：``False``。
        - **turbo** (bool, 可选) - 使用分而治之算法（速度更快，但占用大量内存，仅适用于需要计算全量特征值的广义特征值问题）。
          如果不需要计算特征向量，则没有显著影响。
          默认值：``True``。
        - **eigvals** (tuple, 可选) - 要返回的最小和最大（按升序排列）特征值和对应的特征向量的索引: :math:`0 <= lo <= hi <= M-1`。
          如果缺省，则返回所有特征值和特征向量。
          默认值：``None``。
        - **type** (int, 可选) - 对于广义问题，此参数指定 `w` 和 `v` 要解决的问题类型（仅取1、2、3作为可能的输入）：

          .. code-block::

              1 =>     a @ v = w @ b @ v
              2 => a @ b @ v = w @ v
              3 => b @ a @ v = w @ v

          对于标准问题，会忽略此关键字。默认值：``1``。
        - **check_finite** (bool, 可选) - 是否检查输入矩阵是否只包含有限数。
          禁用可能会带来性能增益，但如果输入确实包含INF或NaN，则可能会导致问题（崩溃、程序不终止）。
          默认值：``True``。

    返回：
        - **w** (Tensor) - 返回shape为 :math:`(N,)` 的Tensor，其中特征值 :math:`N (1<=N<=M)`，按升序排列，根据其多样性重复。

        - **v** (Tensor) - 如果 `eigvals_only==False`，返回shape为 :math:`(M, N)` 的Tensor。

    异常：
        - **RuntimeError** - 如果特征值计算不收敛或 `b` 不是正定矩阵，则会触发报错。
          如果输入矩阵不是对称矩阵或Hermitian矩阵，则不会报告错误，但结果将是错误的。
        - **TypeError** - 如果 `a` 不是Tensor。
        - **TypeError** - 如果 `low` 不是bool类型。
        - **TypeError** - 如果 `eigvals_only` 不是bool类型。
        - **TypeError** - 如果 `overwrite_a` 不是bool类型。
        - **TypeError** - 如果 `overwrite_b` 不是bool类型。
        - **TypeError** - 如果 `turbo` 不是bool类型。
        - **TypeError** - 如果 `check_finite` 不是bool类型。
        - **ValueError** - 如果 `a` 不是2D方阵。
        - **ValueError** - 如果 `b` 不为None。
        - **ValueError** - 如果 `eigvals` 不是None。
