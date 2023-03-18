mindspore.ops.Cross
====================

.. py:class:: mindspore.ops.Cross(dim=-65530)

    返回 `x1` 和 `x2` 沿着维度 `dim` 上的向量积（叉积）。

    更多参考详见 :func:`mindspore.ops.cross` 。

    参数：
        - **dim** (int) - 沿着此维进行叉积操作。默认值：-65530。

    输入：
        - **x1** (Tensor) - 输入Tensor。
        - **x2** (Tensor) - 另一个输入Tensor，数据类型和shape必须和 `x1` 一致，并且他们的 `dim` 维度的长度应该为3。

    输出：
        Tensor，数据类型与输入相同。
