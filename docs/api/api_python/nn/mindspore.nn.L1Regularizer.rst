mindspore.nn.L1Regularizer
===========================

.. py:class:: mindspore.nn.L1Regularizer(scale)

    对权重计算L1正则化的值。

    .. math::
        \text{loss}=\lambda * \text{reduce_sum}(\text{abs}(\omega))

    .. note::
        正则化因子应为大于0。

    **参数：**

    - **scale (int, float)** - L1正则化因子，其值大于0。

    **输入：**

    - **weights** (Tensor)** - L1Regularizer的输入，任意维度的Tensor，数据类型为float16或float32。

    **输出：**

    Tensor，其shape为()，默认数据类型为mindspore.float32，如果权重的数据类型精度更高，则以权重的数据类型作为输出数据类型。

    **异常：**

    - **TypeError** - `scale` 既不是int也不是float。
    - **ValueError** - `scale` 不大于0。
    - **ValueError** - `scale` 是math.inf或math.nan。