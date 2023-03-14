mindspore.ops.mse_loss
======================

.. py:function:: mindspore.ops.mse_loss(input, target, reduction='mean')

    计算预测值和标签值之间的均方误差。

    更多参考详见 :class:`mindspore.nn.MSELoss`。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。
        - **target** (Tensor) - 输入标签，任意维度的Tensor。大多数场景下与 `input` 具有相同的shape。
          但是，也支持在两者shape不相同的情况下，通过广播保持一致。
        - **reduction** (str，可选) - 对loss应用特定的缩减方法。可选"mean"、"none"、"sum"。默认值："mean"。

    返回：
        Tensor，数据类型为float，如果 `reduction` 为 'mean'或'sum'时，shape为0；如果 `reduction` 为 'none'，输入的shape则是广播之后的shape。

    异常：
        - **ValueError** - 如果 `reduction` 的值不是以下其中之一时：'none'、 'mean'、 'sum'。
        - **ValueError** - 如果 `input` 和 `target` 的shape不相同且无法广播。
