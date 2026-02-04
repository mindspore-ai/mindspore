mindspore.mint.nn.functional.cross_entropy
===========================================

.. py:function:: mindspore.mint.nn.functional.cross_entropy(input, target, weight=None, ignore_index=-100, reduction='mean', label_smoothing=0.0)

    获取预测值和目标值之间的交叉熵损失。

    cross_entropy方法支持两种不同的目标值(target)：

    - 类别索引 (int)，取值范围为 :math:`[0, C)` ，其中 :math:`C` 为类别数，当reduction为 ``'none'`` 时，交叉熵损失公式如下：

      .. math::

          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - w_{y_n} \log \frac{\exp(x_{n,y_n})}{\sum_{c=1}^C \exp(x_{n,c})}
          \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}

      其中， :math:`x` 表示预测值， :math:`y` 表示目标值， :math:`w` 表示权重， :math:`N`表示batch size， :math:`c` 限定范围为 :math:`[0, C-1]` ，表示类索引，其中 :math:`C` 表示类的数量。

      若reduction不为 ``'none'`` （默认为 ``'mean'`` ），则

      .. math::

          \ell(x, y) = \begin{cases}
              \sum_{n=1}^N \frac{1}{\sum_{n=1}^N w_{y_n} \cdot \mathbb{1}\{y_n \not= \text{ignore_index}\}} l_n, &
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
              \end{cases}

    - 类别概率 (float)，用于目标值为多个类别标签的情况。当reduction为 ``'none'`` 时，交叉熵损失公式如下：

      .. math::

          \ell(x, y) = L = \{l_1,\dots,l_N\}^\top, \quad
          l_n = - \sum_{c=1}^C w_c \log \frac{\exp(x_{n,c})}{\sum_{i=1}^C \exp(x_{n,i})} y_{n,c}

      其中， :math:`x` 表示预测值， :math:`y` 表示目标值， :math:`w` 表示权重，N表示batch size， :math:`c` 限定范围为 :math:`[0, C-1]` ，表示类索引，其中 :math:`C` 表示类的数量。

      若reduction不为 ``'none'`` （默认为 ``'mean'`` ），则

      .. math::

          \ell(x, y) = \begin{cases}
              \frac{\sum_{n=1}^N l_n}{N}, &
              \text{if reduction} = \text{'mean',}\\
              \sum_{n=1}^N l_n,  &
              \text{if reduction} = \text{'sum'.}
              \end{cases}

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    .. note::
        动态shape、动态rank和可变常量输入不支持在 `严格图模式(jit_syntax_level=mindspore.STRICT)
        <https://www.mindspore.cn/tutorials/zh-CN/master/compile/static_graph.html>`_ 下执行。

    参数：
        - **input** (Tensor) - 输入预测值，shape为 :math:`(N)` 或 :math:`(N, C)` 或 :math:`(N, C, H, W)`
          (针对二维数据)，或 :math:`(N, C, d_1, d_2, ..., d_K)` (针对高维数据)。`input` 需为对数概率。数据类型仅支持float16或float32或bfloat16(仅Atlas A2训练系列产品支持)。
        - **target** (Tensor) - 输入目标值。若目标值为类别索引，则shape为 :math:`()` 、 :math:`(N)` 或 :math:`(N, d_1, d_2, ..., d_K)` ，数据类型仅支持int32或int64。
          若目标值为类别概率，则shape为 :math:`(N,)` 、 :math:`(N, C)` 或 :math:`(N, C, d_1, d_2, ..., d_K)` ，数据类型仅支持float16或float32或bfloat16(仅Atlas A2训练系列产品支持)。
        - **weight** (Tensor, 可选) - 指定各类别的权重。若值不为 ``None`` ，则shape为 :math:`(C,)`。
          数据类型仅支持float16或float32或bfloat16(仅Atlas A2训练系列产品支持)。默认值： ``None`` 。
        - **ignore_index** (int, 可选) - 指定target中需要忽略的值(一般为填充值)，使其不对梯度产生影响。仅在目标值为类别索引下生效，在类别概率下请设置为负数。默认值： ``-100`` 。
        - **reduction** (str, 可选) - 指定应用于输出结果的归约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``''none'``：不应用归约方法。
          - ``'mean'``：计算输出元素的加权平均值。
          - ``'sum'``：计算输出元素的总和。

        - **label_smoothing** (float, 可选) - 标签平滑值，用于计算Loss时防止模型过拟合的正则化手段。取值范围为[0.0, 1.0]。默认值： ``0.0`` 。

    返回：
        Tensor，数据类型与 `input` 相同。
