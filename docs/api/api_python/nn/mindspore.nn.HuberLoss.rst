mindspore.nn.HuberLoss
=============================

.. py:class:: mindspore.nn.HuberLoss(reduction="mean", delta=1.0)

    HuberLoss计算预测值和目标值之间的误差。它兼有L1Loss和MSELoss的优点。

    假设 :math:`x` 和 :math:`y` 为一维Tensor，长度 :math:`N` ，则计算 :math:`x` 和 :math:`y` 的loss而不进行降维操作（即reduction参数设置为"none"）。公式如下：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top

    以及

    .. math::
        l_n = \begin{cases}
            0.5 * (x_n - y_n)^2, & \text{if } |x_n - y_n| < delta; \\
            delta * (|x_n - y_n| - 0.5 * delta), & \text{otherwise. }
        \end{cases}

    其中， :math:`N` 为batch size。如果 `reduction` 不是"none"，则：

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{"mean";}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{"sum".}
        \end{cases}

    参数：
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

        - **delta** (Union[int, float]) - 两种损失之间变化的阈值。该值必须为正。默认值： ``1.0`` 。

    输入：
        - **logits** (Tensor) - 输入预测值，任意维度的Tensor。其数据类型为float16或float32。
        - **labels** (Tensor) - 目标值，通常情况下与 `logits` 的shape和dtype相同。但是如果 `logits` 和 `labels` 的shape不同，需要保证他们之间可以互相广播。

    输出：
        Tensor或Scalar，如果 `reduction` 为"none"，返回与 `logits` 具有相同shape和dtype的Tensor。否则，将返回一个Scalar。

    异常：
        - **TypeError** - `logits` 或 `labels` 的数据类型既不是float16也不是float32。
        - **TypeError** - `logits` 和 `labels` 的数据类型不同。
        - **TypeError** - `delta` 不是float或int。
        - **ValueError** - `delta` 的值小于或等于0。
        - **ValueError** - `reduction` 不为 ``"mean"`` 、 ``"sum"`` 或 ``"none"`` 。
        - **ValueError** - `logits` 和 `labels` 有不同的shape，且不能互相广播。
