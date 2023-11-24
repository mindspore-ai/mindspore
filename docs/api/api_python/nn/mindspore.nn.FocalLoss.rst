mindspore.nn.FocalLoss
=======================

.. py:class:: mindspore.nn.FocalLoss(weight=None, gamma=2.0, reduction='mean')

    FocalLoss函数解决了类别不平衡的问题。

    FocalLoss函数由Kaiming团队在论文 `Focal Loss for Dense Object Detection <https://arxiv.org/pdf/1708.02002.pdf>`_ 中提出，提高了图像目标检测的效果。

    函数如下：

    .. math::
        FL(p_t) = -(1-p_t)^\gamma \log(p_t)

    参数：
        - **gamma** (float) - gamma用于调整Focal Loss的权重曲线的陡峭程度。默认值： ``2.0`` 。
        - **weight** (Union[Tensor, None]) - Focal Loss的权重，维度为1。如果为None，则不使用权重。默认值： ``None`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``'none'``：不应用规约方法。
          - ``'mean'``：计算输出元素的加权平均值。
          - ``'sum'``：计算输出元素的总和。

    输入：
        - **logits** (Tensor) - shape为 :math:`(N, C)` 、 :math:`(N, C, H)` 、或 :math:`(N, C, H, W)` 的Tensor，其中 :math:`C` 是分类的数量，值大于1。如果shape为 :math:`(N, C, H, W)` 或 :math:`(N, C, H)` ，则 :math:`H` 或 :math:`H` 和 :math:`W` 的乘积应与 `labels` 的相同。
        - **labels** (Tensor) - shape为 :math:`(N, C)` 、 :math:`(N, C, H)` 、或 :math:`(N, C, H, W)` 的Tensor， :math:`C` 的值为1，或者与 `logits` 的 :math:`C` 相同。如果 :math:`C` 不为1，则shape应与 `logits` 的shape相同，其中 :math:`C` 是分类的数量。如果shape为 :math:`(N, C, H, W)` 或 :math:`(N, C, H)` ，则 :math:`H` 或 :math:`H` 和 :math:`W` 的乘积应与 `logits` 相同。 `labels` 的值应该在 [-:math:`C`, :math:`C`)范围内，其中 :math:`C` 是logits中类的数量。

    输出：
        Tensor或Scalar，如果 `reduction` 为"none"，其shape与 `logits` 相同。否则，将返回Scalar。

    异常：
        - **TypeError** - `gamma` 的数据类型不是float。
        - **TypeError** - `weight` 不是Tensor。
        - **ValueError** - `labels` 维度与 `logits` 不同。
        - **ValueError** - `labels` 通道不为1，且 `labels` 的shape与 `logits` 不同。
        - **ValueError** - `reduction` 不为 ``'mean'`` ， ``'sum'`` ，或 ``'none'`` 。
