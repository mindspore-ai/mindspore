mindspore.mint.nn.functional.mse_loss
=========================================

.. py:function:: mindspore.mint.nn.functional.mse_loss(input, target, reduction='mean')

    计算预测值和标签值之间的均方误差。

    更多参考详见 :class:`mindspore.mint.nn.MSELoss`。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。数据类型和 `target` 需要一致。
        - **target** (Tensor) - 输入标签，任意维度的Tensor。数据类型和 `input` 需要一致。支持在 `input` 和 `target` shape不相同的情况下，通过广播保持一致。
        - **reduction** (str，可选) - 指定应用于输出结果的归约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``'none'``：不应用归约方法。
          - ``'mean'``：计算输出元素的平均值。
          - ``'sum'``：计算输出元素的总和。

    返回：
        - Tensor。如果 `reduction` 为 ``'mean'`` 或 ``'sum'`` 时，shape为 `Tensor Scalar` 。
        - 如果 `reduction` 为 ``'none'`` ，输出的shape则是 `input` 和 `target` 广播之后的shape。

    异常：
        - **ValueError** - 如果 `reduction` 不为 ``'mean'``， ``'sum'`` 或 ``''none'`` 中的一个。
        - **ValueError** - 如果 `input` 和 `target` 无法广播。
        - **TypeError** - 如果 `input` 和 `target` 数据类型不一致。
