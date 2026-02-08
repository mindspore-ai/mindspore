mindspore.ops.csr_round
========================

.. py:function:: mindspore.ops.csr_round(x)

    对CSRTensor输入数据进行四舍五入到最接近的整数数值。

    .. math::
        out_i \approx x_i

    参数：
        - **x** (CSRTensor) - 输入CSRTensor。

    返回：
        CSRTensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `x` 不是CSRTensor。
