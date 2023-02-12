mindspore.ops.baddbmm
=====================

.. py:function:: mindspore.ops.baddbmm(x, batch1, batch2, beta=1, alpha=1)

    对输入的两个三维矩阵batch1与batch2相乘，并将结果与x相加。
    计算公式定义如下：

    .. math::
        \text{out}_{i} = \beta \text{x}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    参数：
        - **x** (Tensor) - 输入Tensor，shape为 :math:`(C, W, H)` 。
        - **batch1** (Tensor) - 公式中的 :math:`batch1` ，shape为 :math:`(C, W, T)` 。
        - **batch2** (Tensor) - 公式中的 :math:`batch2` ，shape为 :math:`(C, T, H)` 。
        - **beta** (Union[float, int]，可选) - `x` 的系数，默认值为1。
        - **alpha** (Union[float, int]，可选) - :math:`batch1 @ batch2` 的系数，默认值为1。

    返回：
        Tensor，其数据类型与 `x` 相同，其维度与 `batch1@batch2` 的结果相同。

    异常：
        - **TypeError** - `x` 、 `batch1` 或 `batch2` 的类型不是Tensor。
        - **TypeError** - `x` 、 `batch1` 或 `batch2` 数据类型不一致。
        - **TypeError** - 对于输入为浮点类型的Tensor， `beta` 、 `alpha` 不是实数。否则， `beta` 、 `alpha` 不是整数。
        - **TypeError** - `beta` 、 `alpha` 不是实数类型。
        - **ValueError** - `batch1` 或 `batch2` 的不是三维Tensor。
