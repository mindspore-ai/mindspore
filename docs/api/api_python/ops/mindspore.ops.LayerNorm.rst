mindspore.ops.LayerNorm
=======================

.. py:class:: mindspore.ops.LayerNorm(begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7)

    在输入Tensor上应用层归一化（Layer Normalization）。

    此算子将在给定的轴上对输入进行层归一化。`Layer Normalization <https://arxiv.org/abs/1607.06450>`_ 描述了LayerNorm。

    .. math::
        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 是Scalar， :math:`\beta` 是偏置项， :math:`\epsilon` 是精度值。

    参数：
        - **begin_norm_axis** (int) - 指定 `input_x` 需进行层归一化的起始维度，其值必须在[-1, rank(input))范围内。默认值：1。
        - **begin_params_axis** (int) - 指定输入参数(`gamma`, `beta`) 需进行层归一化的开始轴，其值必须在[-1, rank(input))范围内。默认值：1。
        - **epsilon** (float) - 添加到分母中的值，以确保数据稳定性。默认值：1e-7。

    输入：
        - **input_x** (Tensor) - LayerNorm的输入，shape为 :math:`(N, \ldots)` 的Tensor。
        - **gamma** (Tensor) - 可学习参数 :math:`\gamma` ，shape为 :math:`(P_0, \ldots, P_\text{begin_params_axis})` 的Tensor。
        - **beta** (Tensor) - 可学习参数 :math:`\beta` 。shape为 :math:`(P_0, \ldots, P_\text{begin_params_axis})` 的Tensor。

    输出：
        tuple[Tensor]，3个Tensor组成的tuple，层归一化输入和更新后的参数。

        - **output_x** (Tensor) - 层归一化输入，shape为是 :math:`(N, C)` 。数据类型和shape与 `input_x` 相同。
        - **mean** (Tensor) - 输入的均值，shape为 :math:`(C,)` 的Tensor。
        - **variance** (Tensor) - 输入的方差，shape为 :math:`(C,)` 的Tensor。

    异常：
        - **TypeError** - `begin_norm_axis` 或 `begin_params_axis` 不是int。
        - **TypeError** - `epsilon` 不是float。
        - **TypeError** - `input_x`、`gamma` 或 `beta` 不是Tensor。