mindspore.ops.MatrixPower
=========================

.. py:class:: mindspore.ops.MatrixPower(n)

    计算一个batch的方阵的n次幂。
    如果 :math:`n=0` ，则返回一个batch的单位矩阵。
    如果n为负数，则为返回每个矩阵（如果可逆）逆矩阵的 :math:`abs(n)` 次幂。

    参数：
        - **n** (int) - 指数，必须是整数。
  
    输入：
        - **x** (Tensor) - 一个3-D Tensor。支持的数据类型为float16和float32。
          shape为 :math:`(b, m, m)` ，表示b个m-D的方阵。

    输出：
        - **y** (Tensor) - 一个3-D Tensor，与 `x` 的shape和数据类型均相同。

    异常：
        - **TypeError** - 如果 `n` 的数据类型不是整数。
        - **TypeError** - 如果 `x` 的数据类型既不是float16，又不是float32。
        - **TypeError** - 如果 `x` 不是Tensor。
        - **ValueError** - 如果 `x` 不是一个3-D Tensor。
        - **ValueError** - 如果 `x` 的shape[1]和shape[2]不同。
        - **ValueError** - 如果 `n` 为负数，但是输入 `x` 中存在奇异矩阵。
