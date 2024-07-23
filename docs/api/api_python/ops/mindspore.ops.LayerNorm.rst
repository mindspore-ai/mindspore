mindspore.ops.LayerNorm
=======================

.. py:class:: mindspore.ops.LayerNorm(begin_norm_axis=1, begin_params_axis=1, epsilon=1e-7)

    在输入Tensor上应用层归一化（Layer Normalization）。

    此算子将在给定的轴上对输入进行层归一化。`Layer Normalization <https://arxiv.org/abs/1607.06450>`_ 描述了LayerNorm。

    .. math::
        y = \frac{x - mean}{\sqrt{variance + \epsilon}} * \gamma + \beta

    其中 :math:`\gamma` 是Scalar， :math:`\beta` 是偏置项， :math:`\epsilon` 是精度值。

    参数：
        - **begin_norm_axis** (int) - 指定 `input_x` 需进行层归一化的起始维度，其值必须在[-1, rank(input_x))范围内。默认值： ``1`` 。
        - **begin_params_axis** (int) - 指定输入参数(`gamma`, `beta`) 需进行层归一化的开始轴，其值必须在[-1, rank(input_x))范围内。默认值： ``1`` 。注: 在Ascend平台， `begin_params_axis` 需要和 `begin_norm_axis` 的值相等。
        - **epsilon** (float) - 添加到分母中的值（:math:`\epsilon`），以确保数据稳定性。默认值： ``1e-7`` 。

    输入：
        - **input_x** (Tensor) - LayerNorm的输入，shape为 :math:`(N, \ldots)` 的Tensor。支持的数据类型：float16、float32、float64。
        - **gamma** (Tensor) - 可学习参数 :math:`\gamma` ，shape为 `input_x_shape[begin_params_axis:]` 的Tensor。支持的数据类型：float16、float32、float64。
        - **beta** (Tensor) - 可学习参数 :math:`\beta` ，shape为 `input_x_shape[begin_params_axis:]` 的Tensor。支持的数据类型：float16、float32、float64。

    输出：
        tuple[Tensor]，3个Tensor组成的tuple，层归一化输入和更新后的参数。

        - **output_x** (Tensor) - 层归一化输入，数据类型和shape与 `input_x` 相同。
        - **mean** (Tensor) - 输入的均值，其shape的前 `begin_norm_axis` 维与 `input_x` 相同，其余维度为1。假设输入 `input_x` 的shape为 :math:`(x_1, x_2, \ldots, x_R)` , 输出 `mean` 的shape为 :math:`(x_1, \ldots, x_{begin\_norm\_axis}, 1, \ldots, 1)` （当 `begin_norm_axis=0` 时， `mean` shape为 :math:`(1, \ldots, 1)` ）。
        - **rstd** (Tensor) - 输入的标准差的倒数，shape同 `mean` 一致。

    异常：
        - **TypeError** - `begin_norm_axis` 或 `begin_params_axis` 不是int。
        - **TypeError** - `epsilon` 不是float。
        - **TypeError** - `input_x`、`gamma` 或 `beta` 不是Tensor。
