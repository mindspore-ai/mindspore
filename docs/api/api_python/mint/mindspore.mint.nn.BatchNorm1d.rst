mindspore.mint.nn.BatchNorm1d
=============================

.. py:class:: mindspore.mint.nn.BatchNorm1d(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True, dtype=None)

    在二维或三维输入上应用批归一化。具体内容见论文 `Batch Normalization: Accelerating Deep Network Training by
    Reducing Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`_ 。

    .. math::
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    在mini-batch上按维度计算平均值和标准差。:math:`\gamma` 和 :math:`\beta` 是大小为 `C` 的可学习参数向量。
    默认情况下，:math:`\gamma` 中的元素被赋值为1，:math:`\beta` 的元素被赋值为0。

    .. warning::
        该接口不支持动态Rank。

    参数：
        - **num_features** (int) - 输入的shape :math:`(N, C, L)` 中的 `C` 值。
        - **eps** (float, 可选) - :math:`\epsilon` 加在分母上的值，以确保数值稳定。默认 ``1e-5`` 。
        - **momentum** (float, 可选) - 动态均值和动态方差所使用的动量。当想计算累计平均值时，可设置为 ``None``。默认 ``0.1`` 。
        - **affine** (bool, 可选) - bool类型。设置为 ``True`` 时，可学习到仿射参数值。默认 ``True`` 。
        - **track_running_stats** (bool, 可选) - bool类型。设置为 ``True`` 时，会跟踪运行时的均值和方差。当设置为 ``False`` 时，
          则不会跟踪这些统计信息。且在train和eval模式下，该cell总是使用batch的统计信息。默认 ``True`` 。
        - **dtype** (:class:`mindspore.dtype`, 可选) - parameters的dtype。默认 ``None`` 。

    输入：
        - **input** (Tensor) - 输入shape为 :math:`(N, C)` 或 :math:`(N, C, L)` 的Tensor，其中 `N` 为batch，
          `C` 为特征数量或通道数量， `L` 为序列长度。

    输出：
        Tensor，类型和shape与 `input` 一致。

    异常：
        - **ValueError** - `num_features` 小于1。
