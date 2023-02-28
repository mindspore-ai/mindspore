mindspore.nn.Softmax2d
======================

.. py:class:: mindspore.nn.Softmax2d()

    应用于2D特征数据的Softmax函数。

    将 `Softmax` 应用于具有shape :math:`(C, H, W)` 的输入Tensor的每个位置 :math:`(c, h, w)` 。

    输入：
        - **x** (Tensor) - Tensor的shape :math:`(N, C_{in}, H_{in}, W_{in})` 或者 :math:`(C_{in}, H_{in}, W_{in})`。

    输出：
        Tensor，数据类型和shape与 `x` 相同，取值范围为[0, 1]。

    异常：
        - **TypeError** - `x` 的数据类型既不是float16也不是float32。
        - **ValueError** - 数据格式不是“NCHW”或者“CHW”。
