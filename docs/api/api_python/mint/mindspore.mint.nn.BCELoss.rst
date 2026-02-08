mindspore.mint.nn.BCELoss
=========================

.. py:class:: mindspore.mint.nn.BCELoss(weight=None, reduction='mean')

    计算目标值和预测值之间的二值交叉熵损失值。

    将预测值设置为 :math:`x` ，目标值设置为 :math:`y` ，输出损失设置为 :math:`\ell(x, y)` 。

    则公式如下：

    .. math::
        L = \{l_1,\dots,l_n,\dots,l_N\}^\top, \quad
        l_n = - w_n \left[ y_n \cdot \log x_n + (1 - y_n) \cdot \log (1 - x_n) \right]

    其中N是批次大小。公式如下：

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    .. note::
        预测值一般是sigmoid函数的输出。因为是二分类，所以目标值应是0或者1。如果 :math:`x_n` 是0或1，则上述损失函数是无意义的。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **weight** (Tensor, 可选) - 指定每个批次二值交叉熵的权重。与输入数据的shape和数据类型相同。默认 ``None`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的归约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认 ``'mean'`` 。

          - ``"none"``：不应用归约方法。
          - ``"mean"``：计算输出元素的加权平均值。
          - ``"sum"``：计算输出元素的总和。

    输入：
        - **input** (Tensor) - 预测值tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。数据类型必须为mindspore.float16、mindspore.float32或mindspore.bfloat16（仅Atlas A2训练系列产品支持）。
        - **target** (Tensor) - 目标值tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。与 `input` 的shape和数据类型相同。

    输出：
        Tensor，数据类型与 `input` 相同。如果 `reduction` 为 ``'none'`` ，则shape与 `input` 相同。否则，输出为scalar的tensor。

    异常：
        - **ValueError** - `reduction` 不为 ``'none'`` 、 ``'mean'`` 或 ``'sum'`` 。
        - **ValueError** - `input` 的shape与 `target` 或 `weight` （如果给定）不同。
