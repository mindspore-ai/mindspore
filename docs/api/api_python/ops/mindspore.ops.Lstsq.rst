mindspore.ops.Lstsq
===================

.. py:class:: mindspore.ops.Lstsq(fast=True, l2_regularizer=0.0)

    计算满秩矩阵 `x` :math:`(m \times n)` 与满秩矩阵 `a` :math:`(m \times k)` 的最小二乘问题或最小范数问题的解。

    若 :math:`m \geq n` ， `Lstsq` 解决最小二乘问题：
    
    .. math::

       \begin{array}{ll}
       \min_y & \|xy-a\|_2.
       \end{array}

    若 :math:`m < n` ， `Lstsq` 解决最小范数问题：

    .. math::

       \begin{array}{llll}
       \min_y & \|y\|_2 & \text{subject to} & xy = a.
       \end{array}

    参数：
        - **fast** (bool，可选) - 使用的算法。默认值：True。
  
          - 如果 `fast` 为True，则使用Cholesky分解求解正态方程来计算解。
          - 如果 `fast` 为False，则基于数值鲁棒的完全正交分解的算法被使用。
  
        - **l2_regularizer** (float，可选) - L2正则化系数。默认值：0.0。
  
    输入：
        - **x** (Tensor) - :math:`(m \times n)` 的矩阵 `x` 。输入Tensor的数据类型为float16、float32或float64。
        - **a** (Tensor) - :math:`(m \times k)` 的矩阵 `a` 。输入Tensor的数据类型为float16、float32或float64。

    输出：
        Tensor，最小二乘问题或最小范数问题的解，其shape为 :math:`(n \times k)` ，数据类型与 `x` 相同。

    异常：
        - **TypeError** - 若输入 `x` 或 `a` 不是Tensor。
        - **TypeError** - 若 `x` 或 `a` 的数据类型不是以下之一：float16、float32、float64。
        - **TypeError** - 若 `x` 或 `a` 的数据类型不同。
        - **ValueError** - 若 `x` 的维度不等于2。
        - **ValueError** - 若 `a` 的维度不等于2或1。
        - **ValueError** - 若 `x` 与 `a` shape的第零维不相等。
