mindspore.mint.nn.MSELoss
==================================

.. py:class:: mindspore.mint.nn.MSELoss(reduction='mean')

    用于计算预测值与标签值之间的均方误差。

    假设 :math:`x` 和 :math:`y` 为一维Tensor，长度 :math:`N` ，则计算 :math:`x` 和 :math:`y` 的loss（即reduction参数设置为 ``'none'``）的公式如下：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with} \quad l_n = (x_n - y_n)^2.

    其中， :math:`N` 为batch size。如果 `reduction` 不是'none'，则：

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    参数：
        - **reduction** (str，可选) - 指定应用于输出结果的归约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``'none'``：不应用归约方法。
          - ``'mean'``：计算输出元素的平均值。
          - ``'sum'``：计算输出元素的总和。

    输入：
        - **logits** (Tensor) - 输入预测值，任意维度的Tensor。数据类型和 `labels` 需要保持一致。
        - **labels** (Tensor) - 输入标签，任意维度的Tensor。数据类型和 `logits` 需要保持一致。支持在 `logits` 和 `labels` shape不相同的情况下，通过广播保持一致。

    输出：
        - Tensor。如果 `reduction` 为 ``'mean'`` 或 ``'sum'`` 时，输出的shape为 `Tensor Scalar` 。
        - 如果 `reduction` 为 ``'none'`` ，输出的shape则是 `logits` 和 `labels` 广播之后的shape。

    异常：
        - **ValueError** - `reduction` 不为 ``'mean'``， ``'sum'`` 或 ``'none'`` 中的一个。
        - **ValueError** - `logits` 和 `labels` 的shape不能广播。
        - **TypeError** - 如果 `logits` 和 `labels` 数据类型不一致。

