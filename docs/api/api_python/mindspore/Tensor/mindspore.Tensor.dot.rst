mindspore.Tensor.dot
====================

.. py:method:: mindspore.Tensor.dot(other) -> Tensor

    计算两个1D Tensor的点积。

    参数：
        - **other** (Tensor) - 点积的输入，须为1维。

    返回：
        Tensor，shape是[]，类型与 `self` 一致。

    异常：
        - **TypeError** - `other` 的数据类型不是tensor。
        - **RuntimeError** - `self` 和 `other` 的数据类型不一致。
        - **RuntimeError** - `self` 和 `other` 的shape不一致。
        - **RuntimeError** - `self` 或 `other` 不是1D。