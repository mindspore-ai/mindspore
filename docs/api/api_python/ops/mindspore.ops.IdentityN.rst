mindspore.ops.IdentityN
=======================

.. py:class:: mindspore.ops.IdentityN

    返回与输入具有相同shape和值的tuple(Tensor)。

    此操作可用于覆盖复杂函数的梯度。例如，假设 :math:`y = f(x)` ，
    我们希望为反向传播应用自定义函数g，则 :math:`dx=g(dy)` 。

    输入：
        - **x** (Union[tuple[Tensor], list[Tensor]]) - 输入，数据类型为实数。

    输出：
        与输入 `x` 具有相同shape和数据类型的tuple(Tensor)。

    异常：
        - **TypeError** - 如果 `x` 不是tuple(Tensor)或List(Tensor)。
        - **TypeError** - 如果 `x` 的数据类型不是实数。
