mindspore.Tensor.diag
=====================

.. py:method:: mindspore.Tensor.diag() -> Tensor

    详情请参考 :func:`mindspore.ops.diag`。

    .. py:method:: mindspore.Tensor.diag(diagonal=0) -> Tensor
        :noindex:

    如果 `input` 是向量（1-D 张量），则返回一个二维张量，其中 `input` 的元素作为对角线。

    如果 `input` 是矩阵（2-D 张量），则返回具有 `input` 对角线元素的 1-D 张量。

    参数 `diagonal` 控制要考虑的对角线：

    - 如果 `diagonal` = 0，则它是主对角线。

    - 如果 `diagonal` > 0，则它位于主对角线的上方。

    - 如果 `diagonal` < 0，则它位于主对角线下方。

    .. warning::
        - 这是一个实验性API，后续可能修改或删除。
        - 图模式和CPU、GPU后端不支持 `diagonal` 参数取非零值。

    参数：
        - **diagonal** (int, 可选) - 输入Tensor，默认值为 ``0``。

    返回：
        Tensor，具有与输入Tensor相同的数据类型，其shape由 `diagonal` 决定：

        - 如果输入 `input` 的shape为 :math:`(x_0)` ：输出shape为 :math:`(x_0 + \left | diagonal \right | , x_0 + \left | diagonal \right | )` 的二维张量。

        - 如果输入 `input` 的shape为 :math:`(x_0, x_1)` ：输出shape为主对角线上下平移 :math:`(\left | diagonal \right |)` 个单位后所剩元素的长度的一维张量。

    异常：
        - **ValueError** - `input` 的shape不是1-D和2-D。