mindspore.ops.nll_loss
========================

.. py:function:: mindspore.ops.nll_loss(inputs, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)

    获取预测值和目标值之间的负对数似然损失。

    reduction为'none'时，负对数似然损失公式如下：

    .. math::
        \ell(x, t)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top},
        \quad l_{n}=-w_{t_{n}} x_{n, t_{n}},
        \quad w_{c}=\text { weight }[c] \cdot \mathbb{1}
        \{c \not= \text{ignore_index}\},

    其中， :math:`x` 表示预测值， :math:`t` 表示目标值， :math:`w` 表示权重，N表示batch size， :math:`c` 限定范围为[0, C-1]，表示类索引，其中 :math:`C` 表示类的数量。

    若reduction不为'none'（默认为'mean'），则

    .. math::
        \ell(x, t)=\left\{\begin{array}{ll}
        \sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{t n}} l_{n}, & \text { if reduction }=\text { 'mean', } \\
        \sum_{n=1}^{N} l_{n}, & \text { if reduction }=\text { 'sum' }
        \end{array}\right.

    参数：
        - **inputs** (Tensor) - 输入预测值，shape为 :math:`(N, C)` 或 :math:`(N, C, H, W)`
          (针对二维数据), 或 :math:`(N, C, d_1, d_2, ..., d_K)` (针对高维数据)。`inputs` 需为对数概率。数据类型仅支持float32或float16。
        - **target** (Tensor) - 输入目标值，shape为 :math:`(N)` 或 :math:`(N, d_1, d_2, ..., d_K)` (针对高维数据)。
          数据类型仅支持int32。
        - **weight** (Tensor) - 指定各类别的权重。若值不为None，则shape为 (C,)。
          数据类型仅支持float32或float16。默认值：None。
        - **ignore_index** (int) - 指定target中需要忽略的值(一般为填充值)，使其不对梯度产生影响。默认值：-100。
        - **reduction** (str) - 指定应用于输出结果的计算方式，比如'none'、'mean'、'sum'，默认值："mean"。
        - **label_smoothing** (float) - 标签平滑值，用于计算Loss时防止模型过拟合的正则化手段。取值范围为[0.0, 1.0]。默认值：0.0。

    返回：
        Tensor，数据类型与 `inputs` 相同。
