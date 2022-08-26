mindspore.ops.Erfinv
=====================

.. py:class:: mindspore.ops.Erfinv

    计算输入Tensor的逆误差函数。逆误差函数在范围(-1,1)。
    
    公式定义为：

    .. math::
        erfinv(erf(x)) = x

    输入：
        - **input_x** (Tensor) - 待计算的输入Tensor，数据类型为float32或float16。

    输出：
        Tensor，数据类型和shape与 `input_x` 相同。

    异常：
        - **TypeError** - 如果 `input_x` 的数据类型既不是float32也不是float16。
