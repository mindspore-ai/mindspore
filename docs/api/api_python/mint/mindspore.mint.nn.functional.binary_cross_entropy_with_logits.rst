mindspore.mint.nn.functional.binary_cross_entropy_with_logits
=============================================================

.. py:function:: mindspore.mint.nn.functional.binary_cross_entropy_with_logits(input, target, weight=None, reduction='mean', pos_weight=None)

    输入经过sigmoid激活函数后作为预测值， `binary_cross_entropy_with_logits` 计算预测值和目标值之间的二值交叉熵损失。与 `mindspore.ops.binary_cross_entropy_with_logits` 功能一致。

    将输入 `input` 设置为 :math:`X` ，输入 `target` 设置为 :math:`Y` ，输入 `weight` 设置为 :math:`W` ，输出设置为 :math:`L` 。则，

    .. math::
        \begin{array}{ll} \\
            p_{ij} = sigmoid(X_{ij}) = \frac{1}{1 + e^{-X_{ij}}} \\
            L_{ij} = -[Y_{ij}log(p_{ij}) + (1 - Y_{ij})log(1 - p_{ij})]
        \end{array}

    :math:`i` 表示 :math:`i^{th}` 样例， :math:`j` 表示类别。则，

    .. math::
        \ell(x, y) = \begin{cases}
        L, & \text{if reduction} = \text{'none';}\\
        \operatorname{mean}(L), & \text{if reduction} = \text{'mean';}\\
        \operatorname{sum}(L),  & \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`\ell` 表示计算损失的方法。有三种方法：第一种方法是直接提供损失值，第二种方法是计算所有损失的平均值，第三种方法是计算所有损失的总和。

    该算子会将输出乘以相应的权重。
    :math:`weight` 表示一个batch中的每条数据分配不同的权重，
    :math:`pos\_weight` 为每个类别的正例子添加相应的权重。

    此外，它可以通过向正例添加权重来权衡召回率和精度。
    在多标签分类的情况下，损失可以描述为：

    .. math::
        \begin{array}{ll} \\
            p_{ij,c} = sigmoid(X_{ij,c}) = \frac{1}{1 + e^{-X_{ij,c}}} \\
            L_{ij,c} = -[P_{c}Y_{ij,c} * log(p_{ij,c}) + (1 - Y_{ij,c})log(1 - p_{ij,c})]
        \end{array}

    其中 c 是类别数目（c>1 表示多标签二元分类，c=1 表示单标签二元分类），n 是批次中样本的数量，:math:`P_c` 是 第c类正例的权重。
    :math:`P_c>1` 增大召回率, :math:`P_c<1` 增大精度。

    参数：
        - **input** (Tensor) - 输入预测值，shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。其数据类型为float16、float32或bfloat16（仅Atlas A2训练系列产品支持）。
        - **target** (Tensor) - 输入目标值，shape与 `input` 相同。数据类型为float16、float32或bfloat16（仅Atlas A2训练系列产品支持）。
        - **weight** (Tensor, 可选) - 指定每个批次二值交叉熵的权重。支持广播，使其shape与 `target` 的shape保持一致。数据类型必须为float16、float32或bfloat16（仅Atlas A2训练系列产品支持）。默认值：``None`` ， `weight` 是值为 ``1`` 的Tensor。
        - **reduction** (str, 可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``'none'``：不应用规约方法。
          - ``'mean'``：计算输出元素的加权平均值。
          - ``'sum'``：计算输出元素的总和。
        - **pos_weight** (Tensor, 可选) - 指定正类的权重。是一个长度等于分类数的向量。支持广播，使其shape与 `target` 的shape保持一致。数据类型必须为float16、float32或bfloat16（仅Atlas A2训练系列产品支持）。默认值：``None`` ，此时 `pos_weight` 等价于值为 ``1`` 的Tensor。

    返回：
        Tensor或Scalar，如果 `reduction` 为 ``'none'`` ，则为shape和数据类型与输入 `target` 相同的Tensor。否则，输出为Scalar。

    异常：
        - **TypeError** - 输入 `input` ， `target` ， `weight` ， `pos_weight` 不为Tensor。
        - **TypeError** - `reduction` 输入数据类型不为string。
        - **ValueError** - `weight` 或 `pos_weight` 不能广播到shape为 `input` 的Tensor。
        - **ValueError** - `reduction` 不为 ``'none'`` 、 ``'mean'``  或 ``'sum'`` 。

