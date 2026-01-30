mindspore.ops.add_layer_norm
============================

.. py:function:: mindspore.ops.add_layer_norm(x1, x2, gamma, beta, epsilon=1e-5, additional_output=False)

    AddLayerNorm算法的实现。

    .. math::
        \begin{aligned}
        x = x1 + x2                                                                        \\
        y = \frac{x - \mathrm{E}[x]}{\sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta   \\
        \end{aligned}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **x1** (Tensor) - AddLayerNorm中加法计算的输入，将会在算子内做 `x1 + x2` 的计算并对计算结果做层归一化。数据类型 float16、bfloat16、float32。
        - **x2** (Tensor) - AddLayerNorm中加法计算的输入，将会在算子内做 `x1 + x2` 的计算并对计算结果做层归一化。数据类型和shape与 `x1` 相同。
        - **gamma** (Tensor) - 可学习参数 :math:`\gamma`。shape支持 1 维度，数据维度和 `x1` 的尾轴相同。数据类型和 `x1` 相同。
        - **beta** (Tensor) - 可学习参数 :math:`\beta`。shape支持 1 维度，数据维度和 `x1` 的尾轴相同。数据类型和 `x1` 相同。
        - **epsilon** (float, 可选) - 添加到分母中的值，以确保数值稳定。默认 ``1e-5`` 。
        - **additional_output** (bool, 可选) - 表示是否开启 `x=x1+x2` 的输出。默认 ``False`` 。

    返回：
        tuple[Tensor]，4个Tensor组成的tuple，层归一化输出和更新后的参数。

        - **y** (Tensor) - 层归一化输出，数据类型和shape与 `x1` 相同。
        - **mean** (Tensor) - 输入的均值，数据类型和 `x1` 相同。
        - **rstd** (Tensor) - 输入的标准差的倒数，shape同 `mean` 一致。
        - **x** (Tensor) - `x1` + `x2` 的输出。

    异常：
        - **TypeError** - `x1` 不是Tensor。
        - **TypeError** - `x2` 不是Tensor。
        - **TypeError** - `gamma` 不是Tensor。
        - **TypeError** - `beta` 不是Tensor。
        - **TypeError** - `epsilon` 不是float。
        - **TypeError** - `additional_output` 不是bool。