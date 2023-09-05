mindspore.ops.Erfinv
=====================

.. py:class:: mindspore.ops.Erfinv

    计算输入Tensor的逆误差函数。逆误差函数在范围(-1,1)。
    
    更多参考详见 :func:`mindspore.ops.erfinv`。

    输入：
        - **input_x** (Tensor) - 输入Tensor。支持数据类型：

          - Ascend： float16、float32。
          - GPU/CPU： float16、float32、float64。

    输出：
        Tensor，数据类型和shape与 `input_x` 相同。
