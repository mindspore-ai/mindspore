mindspore.ops.logdet
=====================

.. py:function:: mindspore.ops.logdet(input)

    计算方块矩阵或批量方块矩阵的对数行列式。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。

    返回：
        Tensor，`input` 的对数行列式。如果行列式小于0，则返回nan。如果行列式等于0，则返回-inf。

    异常：
        - **TypeError** - 如果 `input` 的dtype不是float32、float64、Complex64或Complex128。
