mindspore.ops.InsertGradientOf
==============================

.. py:class:: mindspore.ops.InsertGradientOf(f)

    为图节点附加回调函数，将在梯度计算时被调用。
    
    参数：
        - **f** (Function) - MindSpore Function。回调函数。

    输入：
        - **input_x** (Any) - 需要附加回调函数的图节点。

    输出：
        Tensor，直接返回输入 `input_x` 。该算子不影响前向计算的结果。

    异常：
        - **TypeError** - `f` 不是MindSpore Function。

