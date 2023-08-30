mindspore.ops.Mish
====================

.. py:class:: mindspore.ops.Mish

    逐元素计算输入Tensor的MISH（Self Regularized Non-Monotonic Neural Activation Function 自正则化非单调神经激活函数）。

    详情请查看 :func:`mindspore.ops.mish` 。

    输入：
        - **x** (Tensor) - 输入Tensor。
          支持数据类型：

          - GPU/CPU：float16、float32、float64。
          - Ascend：float16、float32。

    输出：
        Tensor，与 `x` 的shape和数据类型相同。
