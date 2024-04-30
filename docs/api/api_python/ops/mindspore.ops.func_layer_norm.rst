mindspore.ops.layer_norm
========================

.. py:function:: mindspore.ops.layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5)

    在mini-batch输入上应用层归一化（Layer Normalization）。

    层归一化在递归神经网络中被广泛的应用。适用单个训练用例的mini-batch输入上应用归一化，详见论文 `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_ 。

    与批归一化（Batch Normalization）不同，层归一化在训练和测试时执行完全相同的计算。
    应用于所有通道和像素，即使batch_size=1也适用。公式如下：

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 是通过训练学习出的weight值，:math:`\beta` 是通过训练学习出的bias值。

    参数：
        - **input** (Tensor) - `input` 的shape为 :math:`(N, *)` ， 其中 :math:`*` 表示任意的附加维度。
        - **normalized_shape** (Union(int, tuple[int], list[int])) - 表示需要进行归一化的shape， `normalized_shape` 等于 `input_shape[begin_norm_axis:]` ， `begin_norm_axis` 代表归一化要开始的轴。
        - **weight** (Tensor, 可选) - 可学习的权重值，shape为 `normalized_shape` ，默认值: ``None`` 。为 ``None`` 时，初始化为 ``1`` 。
        - **bias** (Tensor, 可选) - 可学习的偏移值，shape为 `normalized_shape` ，默认值: ``None`` 。为 ``None`` 时，初始化为 ``0`` 。
        - **eps** (float, 可选) - 添加到分母中的值（:math:`\epsilon`），以确保数值稳定。默认值： ``1e-5`` 。

    返回：
        Tensor，归一化后的Tensor，shape和数据类型与 `input` 相同。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `normalized_shape` 既不是list也不是tuple。
        - **TypeError** - `eps` 不是float。