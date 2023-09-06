mindspore.nn.SoftmaxCrossEntropyWithLogits
===========================================

.. py:class:: mindspore.nn.SoftmaxCrossEntropyWithLogits(sparse=False, reduction='none')

    计算预测值与真实值之间的交叉熵。

    使用交叉熵损失函数计算出输入概率（使用softmax函数计算）和真实值之间的误差。

    函数的输入是未标准化的值，表示为 :math:`x` ，格式为 :math:`(N, C)` ，以及相应的目标。

    通常情况下，该函数的输入为各类别的分数值以及对应的目标值，输入格式是  :math:`(N, C)` 。

    对于每个实例 :math:`x_i` ， :math:`i` 的范围为0到N-1，则可得损失为：

    .. math::
        \ell(x_i, c) = - \log\left(\frac{\exp(x_i[c])}{\sum_j \exp(x_i[j])}\right)
        =  -x_i[c] + \log\left(\sum_j \exp(x_i[j])\right)

    其中 :math:`x_i` 是一维的Tensor， :math:`c` 为one-hot中等于1的位置。

    .. note::
        虽然目标值是互斥的，即目标值中只有一个为正，但预测的概率不为互斥。只要求输入的预测概率分布有效。

    参数：
        - **sparse** (bool，可选) - 指定目标值是否使用稀疏格式。默认值： ``False`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'none'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    输入：
        - **logits** (Tensor) - shape :math:`(N, C)` 的Tensor。数据类型为float16或float32。
        - **labels** (Tensor) - shape :math:`(N, )` 的Tensor。如果 `sparse` 为 ``True`` ，则 `labels` 的类型为int32或int64。否则，`labels` 的类型与 `logits` 的类型相同。

    输出：
        Tensor，一个shape和数据类型与logits相同的Tensor。

    异常：
        - **TypeError** - `sparse` 不是bool。
        - **TypeError** - `sparse` 为True，并且 `labels` 的dtype既不是int32，也不是int64。
        - **TypeError** - `sparse` 为False，并且 `labels` 的dtype既不是float16，也不是float32。
        - **ValueError** - `reduction` 不为"mean"、"sum"，或"none"。
