mindspore.scipy.linalg.lstsq
============================

.. py:function:: mindspore.scipy.linalg.lstsq(A, B, rcond=None, driver=None)

    求解计算线性等式系统 :math:`A X = B` 的最小二乘问题 。

    .. note::
        - `lstsq` 目前仅用于 `mindscience` 科学计算场景，尚不支持其他使用场景。
        - `lstsq` 尚不支持Windows平台。

    参数：
        - **A** (Tensor) - 等式左边的左乘Tensor，shape为 :math:`(*, M, N)` ，其中 :math:`*` 表示零或者更多的Batch维度。
        - **B** (Tensor) - 等式右边的Tensor，shape为 :math:`(*, M, K)`，其中 :math:`*` 表示零或者更多的Batch维度。
        - **rcond** (number.Number, 可选) - 在MindSpore中，当前这个参数不起作用，默认值： ``None`` 。
        - **driver** (str, 可选) - 使用哪个LAPACK函数求解最小二乘问题，可选项有
          ``"gels"``, ``"gelsy"``, ``"gelss"``, ``"gelsd"`` ，默认值： ``None`` （ ``"gelsy"`` ）
          如果 `A` 条件数很小， 且 `A` 是一个满秩矩阵，那么 ``"gels"`` 能很好地解决最小二乘问题，如果 `A` 不一定满秩，
          则建议使用 ``"gelsy"``，如果 `A` 的条件数很大， ``"gelsd"`` 能更好地解决该问题， ``"gelss"`` 方法在以前更常用，它\
          占用更少的内存，但是算得更慢。

    返回：
        - **solution** (Tensor)，最小二乘的解，shape为 :math:`(*, N, K)`，其中 :math:`*` 等于广播后的Batch维度。
        - **residues** (Tensor)，:math:`AX - B`结果每一列的第二范数平方，shape为 :math:`(*, K)`，其中 :math:`*` 等于\
          广播后的Batch维度，当 `driver` 是(``"gels"``, ``"gelss"``, ``"gelsd"``)其中之一，且 :math:`M > N` 时才会计算，
          否则返回空Tensor。
        - **rank** (Tensor)， `A` 的有效秩数。shape为 :math:`(*)`，其中 :math:`*` 等于 `A` 的Batch维度。当 `driver` 是
          (``"gelsy"``, ``"gelss"``, ``"gelsd"``)其中之一时才会计算，否则返回空Tensor 。
        - **singular_values** (Tensor)， `A` 的奇异值，shape为 :math:`(*, min(M, N))`，其中 :math:`*` 等于 `A` 的Batch维度。
          当 `driver` 是(``"gelss"``, ``"gelsd"``)其中之一时才会计算，否则返回空Tensor。

    异常：
        - **TypeError** - 如果 `A` 和 `B` 的数据类型不同。
        - **ValueError** - 如果 `A` 的维度小于2。
        - **ValueError** - 如果 `A` 和 `B` 的shape不匹配。
        - **ValueError** - 如果 `driver` 不是 ``None``、 ``"gels"``、 ``"gelsy"``、 ``"gelss"`` 或 ``"gelsd"`` 。