mindspore.ops.Softplus
========================

.. py:class:: mindspore.ops.Softplus

    Softplus激活函数。

    Softplus为ReLU函数的平滑近似。可对一组数值使用来确保转换后输出结果均为正值。函数计算如下：

    .. math::

        \text{output} = \log(1 + \exp(\text{x}))

    输入：
        - **input_x** (Tensor) - 任意维度的输入Tensor。
          支持数据类型：

          - GPU/CPU：float16、float32、float64。
          - Ascend：float16、float32。

    输出：
        Tensor，与 `input_x` 的shape和数据类型相同。

    异常：
        - **TypeError** - `input_x` 不是Tensor。
        - **TypeError** - `input_x` 的数据类型非float16、float32或float64。
