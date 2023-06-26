mindspore.ops.ApplyGradientDescent
===================================

.. py:class:: mindspore.ops.ApplyGradientDescent

    通过从 `var` 中减去 `alpha` * `delta` 来更新 `var` 。

    .. math::
        var = var - \alpha * \delta

    其中 :math:`\alpha` 代表 `alpha` ， :math:`\delta` 代表 `delta` 。

    `var` 和 `delta` 的输入遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为相对最高精度数据类型。

    输入：
        - **var** (Parameter) - 要更新的变量，为任意维度，其数据类型为float32或float16。其shape为 :math:`(N, *)` ，其中 :math:`*` 为任意数量的额外维度。
        - **alpha** (Union[Number, Tensor]) - 调节系数，必须是Scalar。数据类型为float32或float16。
        - **delta** (Tensor) - 变化的Tensor，shape与 `var` 相同。

    输出：
        Tensor，更新后的 `var` 。

    异常：
        - **TypeError** - 如果 `var` 或 `alpha` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `delta` 不是Tensor。
        - **TypeError** - 如果 `alpha` 既不是数值型也不是Tensor。
        - **TypeError** - 如果不支持 `var` 和 `delta` 数据类型转换。
