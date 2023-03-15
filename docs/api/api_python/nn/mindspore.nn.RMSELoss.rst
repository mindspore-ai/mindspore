mindspore.nn.RMSELoss
======================

.. py:class:: mindspore.nn.RMSELoss

    RMSELoss用来测量 :math:`x` 和 :math:`y` 元素之间的均方根误差，其中 :math:`x` 是输入Tensor， :math:`y` 是目标值。

    假设 :math:`x` 和 :math:`y` 为一维Tensor，长度为 :math:`N` ， :math:`x` 和 :math:`y` 的loss为：

    .. math::
        loss = \sqrt{\frac{1}{N}\sum_{i=1}^{N}{(x_i-y_i)^2}}

    输入：
        - **logits** (Tensor) - 输入的预测值Tensor, shape :math:`(N, *)` ，其中 :math:`*` 代表任意数量的附加维度。
        - **labels** (Tensor) - 输入的目标值Tensor，shape :math:`(N, *)` 。一般与 `logits` 的shape相同。如果 `logits` 和 `labels` 的shape不同，需支持广播。

    输出：
        Tensor，输出值为加权损失值，其数据类型为float，其shape为()。
