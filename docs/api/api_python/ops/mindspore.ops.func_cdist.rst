mindspore.ops.cdist
===================

.. py:function:: mindspore.ops.cdist(x1, x2, p=2.0)

    计算两个Tensor每对列向量之间的p-norm距离。

    参数：
        - **x1** (Tensor) - 输入Tensor，shape为 :math:`(B, P, M)` ，B维度为0时该维度被忽略，shape为 :math:`(P, M)` 。在GPU上支持数据类型为[float32, float64]，在CPU上支持[float32]。 
        - **x2** (Tensor) - 输入Tensor，shape为 :math:`(B, R, M)` ，与 `x1` 的数据类型一致。
        - **p** (float，可选) - 计算向量对p-norm距离的P值，P∈[0，∞]。默认值:2.0。

    返回：
        Tensor，p-范数距离，数据类型与 `x1` 一致，shape为(B, P, R)。

    异常：
        - **TypeError** - `x1` 或 `x2` 不是Tensor。
        - **TypeError** - `x1` 或 `x2` 的数据类型在GPU环境不是float32或者float64，在CPU环境不是float32。
        - **TypeError** - `p` 不是float32。
        - **ValueError** - `p` 是负数。
        - **ValueError** - `x1` 与 `x2` 维度不同。
        - **ValueError** - `x1` 与 `x2` 的维度既不是2，也不是3。
        - **ValueError** - 单批次训练下 `x1` 和 `x2` 的shape不一样。
        - **ValueError** - `x1` 和 `x2` 的列数不一样。
