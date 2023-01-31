mindspore.ops.cartesian_prod
=============================

.. py:function:: mindspore.ops.cartesian_prod(inputs)

    对给定Tensor序列计算Cartesian乘积，类似于Python里的 `itertools.product` 。

    参数：
        - **inputs** (List[Tensor]) - Tensor序列。

    返回：
        Tensor，Tensor序列的Cartesian乘积。

    异常：
        - **TypeError** - 输入不是Tensor类型。
