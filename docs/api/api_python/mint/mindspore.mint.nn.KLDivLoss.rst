mindspore.mint.nn.KLDivLoss
===========================

.. py:class:: mindspore.mint.nn.KLDivLoss(reduction='mean', log_target=False)

    计算输入 `input` 和 `target` 的Kullback-Leibler散度。

    对于相同shape的Tensor :math:`x` 和 :math:`y` ，KLDivLoss的计算公式如下：

    .. math::
        L(x, y) = y \cdot (\log y - x)

    则

    .. math::
        \ell(x, y) = \begin{cases}
        L(x, y), & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L(x, y)), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L(x, y)) / x.\operatorname{shape}[0], & \text{if reduction} = \text{'batchmean';}\\
        \operatorname{sum}(L(x, y)),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    其中 :math:`x` 表示 `input` ， :math:`y` 表示 `target` ， :math:`\ell(x, y)` 表示输出。

    .. note::
        仅当 `reduction` 设置为 ``"batchmean"`` 时，输出才与Kullback-Leibler散度的数学定义一致。

    参数：
        - **reduction** (str，可选) - 指定应用于输出结果的归约计算方式。默认值： ``'mean'`` 。
        - **log_target** (bool，可选) - 指定输入 `target` 是否在对数空间内。默认值： ``False`` 。

    输入：
        - **input** (Tensor) - 输入Tensor。数据类型必须为float16、float32或bfloat16（仅Atlas A2训练系列产品支持）。
        - **target** (Tensor) - 目标Tensor，其数据类型与 `input` 相同。 `target` 的shape和 `input` 的shape能广播。

    输出：
        Tensor，数据类型与 `input` 相同。如果 `reduction` 为 ``'none'`` ，则shape与 `input` 和 `target` 广播之后的结果相同。
        否则，输出为Scalar的Tensor。

    异常：
        - **TypeError** - `input` 或 `target` 不是Tensor。
        - **TypeError** - `input` 或 `target` 的数据类型不是float16、float32或bfloat16。
        - **TypeError** - `target` 的数据类型与 `input` 不同。
        - **ValueError** - `reduction` 不为 ``'none'`` 、 ``'mean'``、 ``'sum'`` 或 ``'batchmean'`` 之一。
        - **ValueError** - `target` 的shape和 `input` 的shape不能广播。
