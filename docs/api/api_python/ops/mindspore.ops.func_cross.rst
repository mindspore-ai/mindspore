mindspore.ops.cross
====================

.. py:function:: mindspore.ops.cross(input, other, dim=None)

    返回沿着维度 `dim` 上，`input` 和 `other` 的向量积（叉积）。 `input` 和 `other` 必须有相同的形状，且指定的 `dim` 维上size必须为3。
    如果不指定 `dim`，则默认为第一维为3。
    
    参数：
        - **input** (Tensor) - 输入Tensor。
        - **other** (Tensor) - 另一个Tensor，数据类型和shape必须和 `input` 一致，并且他们的 `dim` 维度的长度应该为3。
        - **dim** (int) - 沿着此维进行叉积操作。如果 `dim` 为None，则使用大小为3的第一个维度。默认值：None。

    返回：
        Tensor，数据类型与 `input` 相同。

    异常：
        - **TypeError** - 如果 `input` 不是Tensor。
        - **TypeError** - 如果 `other` 不是Tensor。
        - **TypeError** - 如果 `input` 数据类型与 `other` 不同。
        - **ValueError** - 如果 `input` 和 `other` 的size不同，维度不为3。
        - **ValueError** - 如果 `input` 和 `other` 的shape不相同。
        - **ValueError** - 如果 `dim` 不在[-len(input.shape), len(input.shape)-1]范围内。
