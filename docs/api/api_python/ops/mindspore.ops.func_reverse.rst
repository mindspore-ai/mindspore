mindspore.ops.reverse
==============================

.. py:function:: mindspore.ops.reverse(x, axis)

    对输入Tensor按指定维度反转。

    .. warning::
        "axis"的取值范围为[-dims, dims - 1]，"dims"表示"x"的维度长度。

    参数：
        - **x** (Tensor) - 输入需反转的任意维度的Tensor。数据类型为数值型，不包括float64。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **axis** (Union[tuple(int), list(int)]) - 指定反转的轴。

    输出：
        Tensor，shape和数据类型与输入 `x` 相同。

    异常：
        - **TypeError** - `axis` 既不是list也不是tuple。
        - **TypeError** - `axis` 的元素不是int。
