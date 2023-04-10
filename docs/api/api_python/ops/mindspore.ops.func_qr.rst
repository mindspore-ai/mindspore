mindspore.ops.qr
================

.. py:function:: mindspore.ops.qr(input, mode='reduced')

    返回一个或多个矩阵的QR（正交三角）分解。如果 `mode` 被设为'reduced'(默认值)，则计算q的P列，其中P是 `input` 的2个最内层维度中的最小值。如果 `some` 被设为'complete'，则计算全尺寸q和r。

    参数：
        - **input** (Tensor) - 要进行分解的矩阵。矩阵必须至少为二维。数据类型：float16、float32、float64、complex64、complex128。
          将 `input` 的shape定义为 :math:`(..., m, n)` ，p定义为m和n的最小值。
        - **mode** (Union['reduced', 'complete'], 可选) - 如果 `mode` 的值为'reduced', 则进行部分尺寸的QR分解，否则进行全尺寸的QR
          分解。默认值：'reduced'。

    返回：
        - **Q** (Tensor) - `input` 的正交矩阵。如果 `mode` 为'complete'，则shape为 :math:`(m, m)` ，
          否则shape为 :math:`(m, p)` 。 `Q` 的数据类型与 `input` 相同。
        - **R** (Tensor) - `input` 的上三角形矩阵。如果 `mode` 为'complete'，则shape为 :math:`(m, n)` ，
          否则shape为 :math:`(p, n)` 。 `R` 的数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `mode` 既不是'reduced'也不是'complete'。
        - **ValueError** - `input` 的维度小于2。