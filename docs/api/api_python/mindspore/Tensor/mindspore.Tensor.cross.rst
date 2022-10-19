mindspore.Tensor.cross
======================

.. py:method:: mindspore.Tensor.cross(other, dim=None)

    返回沿着维度 `dim` 上，当前Tensor和 `other` 的向量积（叉积）。当前Tensor和 `other` 必须有相同的形状，且指定的 `dim` 维上size必须为3。
    如果不指定 `dim`，则默认为第一维为3。

    参数：
        - **other** (Tensor) - 输入Tensor。
        - **dim** (int) - 沿着此维进行叉积操作。默认值：None。

    返回：
        Tensor，数据类型与当前Tensor相同。

    异常：
        - **TypeError** - 如果 `other` 不是Tensor。
        - **TypeError** - 如果当前Tensor的数据类型与 `other` 不同。
        - **ValueError** - 如果当前Tensor和 `other` 的size不同，维度不为3。
        - **ValueError** - 如果当前Tensor和 `other` 的shape不相同。
        - **ValueError** - 如果 `dim` 不在[-len(input.shape), len(input.shape)-1]范围内。
