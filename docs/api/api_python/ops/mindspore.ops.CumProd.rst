mindspore.ops.CumProd
=====================

.. py:class:: mindspore.ops.CumProd(exclusive=False, reverse=False)

    计算 `x` 沿着指定axis的元素累计积。

    例如，如果输入是长度为N的vector，则输出也是长度为N的vector。其中的每一个元素为

    .. math::
        y_i = x_1 * x_2 * x_3 * ... * x_i

    参数：
        - **exclusive** (bool) - 如果为True，则排除末尾元素计算元素累计积（见示例）。默认值：False。
        - **reverse** (bool) - 如果为True，则沿 `axis` 反转结果。默认值：False。

    输入：
        - **x** (Tensor[Number]) - 输入Tensor，shape: :math:`(N,*)` ，其中 :math:`*` 表示任何数量的附加维度。其秩应小于8。
        - **axis** (int) - 沿此方向计算累计积。仅支持常量值。

    输出：
        Tensor，shape和数据类型与 `x` 相同。

    异常：
        - **TypeError** - `exclusive` 或 `reverse` 不是bool类型。
        - **TypeError** - `axis` 不是int。
        - **ValueError** - `axis` 不是None。
