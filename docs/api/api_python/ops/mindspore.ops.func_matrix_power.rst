mindspore.ops.matrix_power
==========================

.. py:function:: mindspore.ops.matrix_power(x, n)

    计算一个方阵的（整数）n次幂。
    如果 :math:`n=0` ，则返回一个单位矩阵，其shape和输入 `x` 一致。
    如果 :math:`n<0` 且输入矩阵 `x` 可逆，则返回输入矩阵 `x` 的逆矩阵的 :math:`-n` 次幂。

    参数：
        - **x** (Tensor) - 一个3-D Tensor。支持的数据类型为float16和float32。
          shape为 :math:`(b, m, m)` ，表示b个m-D的方阵。
        - **n** (int) - 指数，必须是整数。

    返回：
        一个3-D Tensor，与 `x` 的shape和数据类型均相同。

    异常：
        - **TypeError** - 如果 `n` 的数据类型不是整数。
        - **TypeError** - 如果 `x` 的数据类型既不是float16，又不是float32。
        - **TypeError** - 如果 `x` 不是Tensor。
        - **ValueError** - 如果 `x` 不是一个3-D Tensor。
        - **ValueError** - 如果 `x` 的shape[1]和shape[2]不同。
        - **ValueError** - 如果 `n` 为负数，但是输入 `x` 中存在奇异矩阵。
