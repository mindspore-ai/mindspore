mindspore.ops.Neg
===================

.. py:class:: mindspore.ops.Neg

    计算输入x的相反数并返回。

    .. math::
        out_{i} = - x_{i}

    **输入：**
 
    - **x** (Tensor) - Neg的输入，任意维度的Tensor，秩应小于8。其数据类型为number。

    **输出：**

    Tensor，shape和类型与输入相同。

    **异常：**

    - **TypeError** - `x` 不是Tensor。