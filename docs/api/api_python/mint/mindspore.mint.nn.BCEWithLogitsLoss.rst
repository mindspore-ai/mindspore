mindspore.mint.nn.BCEWithLogitsLoss
===================================

.. py:class:: mindspore.mint.nn.BCEWithLogitsLoss(weight=None, reduction='mean', pos_weight=None)

    输入 `input` 经过sigmoid激活函数后作为预测值， 用此预测值计算和目标值之间的二值交叉熵损失。

    将输入 `input` 设置为 :math:`X`，输入 `target` 为 :math:`Y`，输出为 :math:`L`。则公式如下：

    .. math::
        p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}}

    .. math::
        L_{ij} = -[Y_{ij} \cdot \log(p_{ij}) + (1 - Y_{ij}) \cdot \log(1 - p_{ij})]

    然后，

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    参数：
        - **weight** (Tensor, 可选) - 指定每个批次二值交叉熵的权重。如果不是None，其shape需要能广播到与 `target` 的shape保持一致，数据类型必须为float16、float32或bfloat16（仅Atlas A2训练系列产品支持）。默认值： ``None`` 。
        - **reduction** (str, 可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``'none'``：不应用规约方法。
          - ``'mean'``：计算输出元素的加权平均值。
          - ``'sum'``：计算输出元素的总和。

        - **pos_weight** (Tensor, 可选) - 指定正样本的权重。是一个长度等于分类数的向量。如果不是None，其shape需要能广播到与 `target` 的shape保持一致，数据类型必须为float16、float32或bfloat16（仅Atlas A2训练系列产品支持）。默认值： ``None`` 。

    输入：
        - **input** (Tensor) - 输入预测值Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。其数据类型为float16、float32或bfloat16（仅Atlas A2训练系列产品支持）。
        - **target** (Tensor) - 输入目标值Tensor，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。与 `input` 的shape和数据类型相同。

    输出：
        Tensor或Scalar，如果 `reduction` 为 ``'none'`` ，其shape需和 `input` 相同。否则，将返回Scalar。

    异常：
        - **TypeError** - `input` 或 `target` 的不为Tensor。
        - **TypeError** - `weight` 或 `pos_weight` 是 Parameter。
        - **TypeError** - `reduction` 的数据类型不是string。
        - **ValueError** - `weight` 或 `pos_weight` 不能广播到shape为 `input` 的Tensor。
        - **ValueError** - `reduction` 不为 ``'none'`` 、 ``'mean'`` 或 ``'sum'``。


