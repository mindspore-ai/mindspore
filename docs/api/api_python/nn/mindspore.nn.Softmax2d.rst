mindspore.nn.Softmax2d
======================

.. py:class:: mindspore.nn.Softmax2d()

    将 SoftMax 应用于每个空间位置的特征。
    当给定Channels x Height x Width的Tensor时，它将 `Softmax` 应用于每个位置 :math:`(Channels, h_i, w_j)`。

    输入：
        - **x** (Tensor) - Tensor的shape :math:`(N, C_{in}, H_{in}, W_{in})` 或者 :math:`(C_{in}, H_{in}, W_{in})`。

    输出：
        Tensor，数据类型和shape与 `x` 相同，取值范围为[0, 1]。

    异常：
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **ValueError** - 数据格式不是“NCHW”或者“CHW”。
