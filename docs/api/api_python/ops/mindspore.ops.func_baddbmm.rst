mindspore.ops.baddbmm
=====================

.. py:function:: mindspore.ops.baddbmm(input, batch1, batch2, beta=1, alpha=1)

    对输入的两个三维矩阵batch1与batch2相乘，并将结果与input相加。
    计算公式定义如下：

    .. math::
        \text{out}_{i} = \beta \text{input}_{i} + \alpha (\text{batch1}_{i} \mathbin{@} \text{batch2}_{i})

    参数：
        - **input** (Tensor) - 输入Tensor，shape为 :math:`(C, W, H)` 。
        - **batch1** (Tensor) - 公式中的 :math:`batch1` ，shape为 :math:`(C, W, T)` 。
        - **batch2** (Tensor) - 公式中的 :math:`batch2` ，shape为 :math:`(C, T, H)` 。
        - **beta** (Union[float, int]，可选) - `input` 的系数，默认值为1。
        - **alpha** (Union[float, int]，可选) - :math:`batch1 @ batch2` 的系数，默认值为1。

    返回：
        Tensor，其数据类型与 `input` 相同，其维度与 `batch1@batch2` 的结果相同。

    异常：
        - **TypeError** - `input` 、 `batch1` 或 `batch2` 的类型不是Tensor。
        - **TypeError** - `input` 、 `batch1` 或 `batch2` 数据类型不一致。
        - **TypeError** - 对于输入为浮点类型的Tensor， `beta` 、 `alpha` 不是实数。否则， `beta` 、 `alpha` 不是整数。
        - **TypeError** - `beta` 、 `alpha` 不是实数类型。
        - **ValueError** - `batch1` 或 `batch2` 的不是三维Tensor。
