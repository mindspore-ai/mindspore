mindspore.nn.GaussianNLLLoss
=============================

.. py:class:: mindspore.nn.GaussianNLLLoss(*, full=False, eps=1e-6, reduction='mean')

    服从高斯分布的负对数似然损失。

    目标值被认为是高斯分布的采样，其中期望和方差通过神经网络来预测。对于以高斯分布为模型的张量 `labels` 和记录期望的张量 `logits` ，以及均为正数的方差张量 `var` 来说，计算的loss为：

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{logits} - \text{labels}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    其中，:math:`eps` 用于 :math:`log` 的稳定性。当 :math:`full=True` 时，一个常数会被添加到loss中。如果 :math:`var` 和 :math:`logits` 的shape不一致（出于同方差性的假设），那么它们必须能够正确地广播。

    参数：
        - **full** (bool) - 指定损失函数中的常数部分。如果为True，则常数为 :math:`const = 0.5*log(2*pi)`。默认值：False。
        - **eps** (float) - 用于提高log的稳定性。默认值：1e-6。
        - **reduction** (str) - 指定应用于输出结果的计算方式，'none'、'mean'、'sum'，默认值：'mean'。

    输入：
        - **logits** (Tensor) - shape为 :math:`(N, *)` 或 :math:`(*)`。`*` 代表着任意数量的额外维度。
        - **labels** (Tensor) - shape为 :math:`(N, *)` 或 :math:`(*)`。和 `logits` 具有相同shape，或者相同shape但有一个维度为1（以允许广播）。
        - **var** (Tensor) - shape为 :math:`(N, *)` 或 :math:`(*)`。和 `logits` 具有相同shape，或者相同shape但有一个维度为1，或者少一个维度（以允许广播）。

    返回：
        Tensor或Tensor scalar，根据 :math:`reduction` 计算的loss。

    异常：
        - **TypeError** - `logits` 不是Tensor。
        - **TypeError** - `labels` 不是Tensor。
        - **TypeError** - `full` 不是bool。
        - **TypeError** - `eps` 不是float。
        - **ValueError** - `eps` 不是在[0, inf)区间的float。
        - **ValueError** - `reduction` 不是"none"、"mean"或者"sum"。
