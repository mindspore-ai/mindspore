mindspore.mint.nn.NLLLoss
==========================

.. py:class:: mindspore.mint.nn.NLLLoss(weight=None, ignore_index=-100, reduction='mean')

    获取预测值和目标值之间的负对数似然损失。

    reduction为 ``'none'`` 时，负对数似然损失公式如下：

    .. math::
        \ell(x, t)=L=\left\{l_{1}, \ldots, l_{N}\right\}^{\top},
        \quad l_{n}=-w_{t_{n}} x_{n, t_{n}},
        \quad w_{c}=\text { weight }[c] \cdot \mathbb{1}
        \{c \not= \text{ignore_index}\},

    其中， :math:`x` 表示预测值， :math:`t` 表示目标值， :math:`w` 表示权重， :math:`N` 表示batch size， :math:`c` 限定范围为 :math:`[0, C-1]`，表示类索引，其中 :math:`C` 表示类的数量。

    若reduction不为 ``'none'`` （默认为 ``'mean'`` ），则

    .. math::
        \ell(x, t)=\left\{\begin{array}{ll}
        \sum_{n=1}^{N} \frac{1}{\sum_{n=1}^{N} w_{t n}} l_{n}, & \text { if reduction }=\text { 'mean', } \\
        \sum_{n=1}^{N} l_{n}, & \text { if reduction }=\text { 'sum' }
        \end{array}\right.

    参数：
        - **weight** (Tensor, 可选) - 指定各类别的权重。若值不为 ``None`` ，则shape为 :math:`(C,)`。
          数据类型仅支持float16、float32或bfloat16(仅Atlas A2训练系列产品支持)。要求与 `input` 的数据类型保持一致。默认值： ``None`` 。
        - **ignore_index** (int, 可选) - 指定target中需要忽略的值(一般为填充值)，使其不对梯度产生影响。仅在目标值为类别索引下生效，在类别概率下请设置为负数。默认值： ``-100`` 。
        - **reduction** (str, 可选) - 指定应用于输出结果的归约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``'none'``：不应用归约方法。
          - ``'mean'``：计算输出元素的加权平均值。
          - ``'sum'``：计算输出元素的总和。

    输入：
        - **input** (Tensor) - 输入预测值，shape为 :math:`(N)` 或 :math:`(N, C)` ， :math:`C` 表示类的数量， :math:`N` 表示batch size，
          或 :math:`(N, C, d_1, d_2, ..., d_K)` (针对高维数据)。`input` 需为对数概率。数据类型仅支持float16、float32或bfloat16(仅Atlas A2训练系列产品支持)。
        - **target** (Tensor) - 输入目标值，shape为 :math:`()` 或 :math:`(N)` ，限定范围为 :math:`[0, C-1]`，或 :math:`(N, d_1, d_2, ..., d_K)` (针对高维数据)。
          数据类型支持int32、int64、uint8。

    输出：
        Tensor，数据类型与 `input` 相同。