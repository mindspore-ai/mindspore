mindspore.mint.nn.functional.l1_loss
====================================

.. py:function:: mindspore.mint.nn.functional.l1_loss(input, target, reduction='mean')

    用于计算预测值和目标值之间的平均绝对误差。

    假设 :math:`x` 和 :math:`y` 为预测值和目标值，均为一维Tensor，长度 :math:`N` ， `reduction` 设置为 ``'none'`` ，则计算 :math:`x` 和 :math:`y` 的loss不进行降维操作。

    公式如下：

    .. math::
        \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad \text{with } l_n = \left| x_n - y_n \right|,

    其中， :math:`N` 为batch size。

    如果 `reduction` 是 ``'mean'`` 或者 ``'sum'`` ，则：

    .. math::
        \ell(x, y) =
        \begin{cases}
            \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    参数：
        - **input** (Tensor) - 预测值，任意维度的Tensor。
        - **target** (Tensor) - 目标值，通常情况与 `input` 的shape相同。如果 `input` 和 `target` 的shape不同，需要保证他们之间可以互相广播。
        - **reduction** (str,可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``'none'``：不应用规约方法。
          - ``'mean'``：计算输出元素的平均值。
          - ``'sum'``：计算输出元素的总和。

    返回：
        Tensor或Scalar，如果 `reduction` 为 ``''none'`` ，则返回与 `input` 具有相同shape和dtype的Tensor。否则，将返回Scalar。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **TypeError** - `target` 不是Tensor。
        - **ValueError** - `reduction` 不为 ``'none'`` 、 ``'mean'`` 或 ``'sum'`` 。
