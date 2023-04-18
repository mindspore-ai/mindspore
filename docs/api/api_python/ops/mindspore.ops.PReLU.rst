mindspore.ops.PReLU
===================

.. py:class:: mindspore.ops.PReLU

    带参数的线性修正单元激活函数（Parametric Rectified Linear Unit activation function）。

    更多参考详见 :func:`mindspore.ops.prelu`。

    输入：
        - **x** (Tensor) - 激活函数的输入Tensor。数据类型为float16或float32。shape为 :math:`(N, C, *)` ，其中 :math:`*` 表示任意的附加维度。
        - **weight** (Tensor) - 权重Tensor。数据类型为float16或float32。`weight` 只可以是向量，其长度与输入 `x` 的通道数C相同。在GPU设备上，当输入为标量时，shape为1。

    输出：
        Tensor，数据类型与 `x` 的相同。
