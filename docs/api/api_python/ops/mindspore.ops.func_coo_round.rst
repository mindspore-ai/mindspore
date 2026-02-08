mindspore.ops.coo_round
========================

.. py:function:: mindspore.ops.coo_round(x)

    对COOTensor输入数据进行四舍五入到最接近的整数数值。

    .. math::
        out_i \approx x_i

    参数：
        - **x** (COOTensor) - 输入COOTensor。

    返回：
        COOTensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `x` 不是COOTensor。
