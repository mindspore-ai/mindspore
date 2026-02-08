mindspore.mint.nn.LayerNorm
===========================

.. py:class:: mindspore.mint.nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, bias=True, dtype=None)

    对mini-batch输入进行层归一化（Layer Normalization）。

    层归一化对单个训练用例的mini-batch输入进行归一化，详见论文 `Layer Normalization <https://arxiv.org/pdf/1607.06450.pdf>`_ 。

    不同于批量归一化和实例归一化，它们对每个通道应用标量缩放和偏置。每个完整的通道/平面使用仿射选项时，层归一化会对每个元素进行缩放
    和偏置项一起使用，进行元素仿射变换。其中 :math:`\gamma` 是通过训练学习出的scale值，:math:`\beta` 是通过训练学习出的shift值。公式如下：

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **normalized_shape** (Union(tuple[int], list[int], int)) - 对 `x` 执行归一化的shape。
        - **eps** (float，可选) - 添加到分母中的值（:math:`\epsilon`），以确保数值稳定。默认 ``1e-5`` 。
        - **elementwise_affine** (bool，可选) - 是否需要仿射变换。一个布尔值，当设置为 ``True`` 时，该模具有可学习的每元素仿射参数，
          初始化为 1（权重）和 0（偏置）。默认 ``True`` 。
        - **bias** (bool，可选) - 当被设置为 ``False`` 时，不会学习结果性偏差（仅 `elementwise_affine` 值为 ``True`` 时生效）。默认 ``True`` 。
        - **dtype** (:class:`mindspore.dtype`，可选) - Parameters的dtype。默认 ``None`` 。

    输入：
        - **x** (Tensor) - `x` 的最后几个维度需要与 `normalized_shape` 相对应，其shape一般为 :math:`(N, ..., *normalized\_shape)`。

    输出：
        - **output** (Tensor) - 归一化后的Tensor，shape和数据类型与 `x` 相同。