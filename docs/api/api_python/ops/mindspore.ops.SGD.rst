mindspore.ops.SGD
=================

.. py:class:: mindspore.ops.SGD(dampening=0.0, weight_decay=0.0, nesterov=False)

    计算随机梯度下降。动量为可选。

    Nesterov动量基于论文 `On the importance of initialization and momentum in deep learning <http://proceings.mlr.press/v28/sutskever13.html>`_ 里的公式。

    .. note::
        如果参数没有分组，优化器中的 `weight_decay` 将应用于名称中没有'beta'或'gamma'的网络参数。用户可以将参数分组来改变权重衰减的策略。当参数被分组时，每个组可以设置 `weight_decay`。如果没有，优化器中的 `weight_decay` 将被应用。
        有关更多详细信息，请参阅: :class:`mindspore.nn.SGD` 。

    参数：
        - **dampening** (float) - 动量的抑制因子。默认值：0.0。
        - **weight_decay** (float) - 权重衰减系数（L2惩罚）。默认值：0.0。
        - **nesterov** (bool) - 是否启用Nesterov动量。默认值：False。

    输入：
        - **parameters** (Tensor) - 待更新的参数。数据类型为float16或float32。
        - **gradient** (Tensor) - 梯度，数据类型为float16或float32。
        - **learning_rate** (Tensor) - 学习率，是一个Scalar的Tensor，数据类型为float16或float32。例如Tensor(0.1, mindspore.float32)。
        - **accum** (Tensor) - 待更新的累加器（速度）。数据类型为float16或float32。
        - **momentum** (Tensor) - 动量，是一个Scalar的Tensor，数据类型为float16或float32。例如Tensor(0.1, mindspore.float32)。
        - **stat** (Tensor) - 待更新的状态，其shape与梯度相同，数据类型为float16或float32。

    输出：
        Tensor，更新后的参数。

    异常：
        - **TypeError** - `dampening` 或 `weight_decay` 不是float。
        - **TypeError** - `nesterov` 不是bool。
        - **TypeError** - `parameters` 、 `gradient` 、 `learning_rate` 、 `accum` 、 `momentum` 或 `stat` 不是Tensor。
        - **TypeError** - `parameters` 、 `gradient` 、 `learning_rate` 、 `accum` 、 `momentum` 或 `stat` 的数据类型既不是float16也不是float32。
