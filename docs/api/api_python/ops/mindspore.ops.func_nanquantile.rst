mindspore.ops.nanquantile
=========================

.. py:function:: mindspore.ops.nanquantile(input, q, axis=None, keepdims=False)

    此接口为mindspore.ops.quantile()的变体，计算的时候会忽略NaN值。如果进行规约计算维度的所有值都是NaN，则返回的分位数为NaN。

    详情请参考 :func:`mindspore.ops.quantile` 。

    参数：
        - **input** (Tensor) - 输入Tensor。其shape为 :math:`(x_1, x_2, ..., x_R)` 。支持的数据类型为：float32、float64。
        - **q** (Union[float, Tensor]) - 标量或1D Tensor。其值范围在[0, 1]，支持的数据类型为：float32、float64。
        - **axis** (int，可选) - 要减少的维度。默认情况下 `axis` 为None，导致输入Tensor在计算前被展平。默认值：None。
        - **keepdims** (bool，可选) - 输出Tensor是否保留维度。默认值：False。

    返回：
        输入Tensor，数据类型与 `input` 一致。

        假设 `input` 的shape为 :math:`(m, x_0, x_1, ..., x_i, ..., X_R)` ， `axis` = :math:`i` ，m为 `q` 中的总元素个数。

        - 如果 `q` 为标量且 `keepdims` 为True，则输出shape为 :math:`(x_0, x_1, ..., 1, ..., X_R)` 。
        - 如果 `q` 为标量且 `keepdims` 为False，则输出shape为 :math:`(x_0, x_1, ..., X_R)` 。
        - 如果 `q` 为1D Tensor且 `keepdims` 为True，则输出shape为 :math:`(m, x_0, x_1, ..., 1, ..., X_R)` 。
        - 如果 `q` 为1D Tensor且 `keepdims` 为False，则输出shape为 :math:`(m, x_0, x_1, ..., X_R)` 。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `q` 不是Tensor或float类型。
        - **TypeError** - `input` 的数据类型不是float32或float64。
        - **TypeError** - `q` 的数据类型不是float32或float64。
        - **TypeError** - `input` 和 `q` 的数据类型不一致。
        - **ValueError** - `q` 的值不在[0, 1]范围内。
        - **ValueError** - `axis` 的值不在有效范围内。
