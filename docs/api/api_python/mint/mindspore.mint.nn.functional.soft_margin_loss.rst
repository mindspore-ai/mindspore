mindspore.mint.nn.functional.soft_margin_loss
===============================================

.. py:function:: mindspore.mint.nn.functional.soft_margin_loss(input, target, reduction='mean')

    计算 `input` 和 `target` 之间的soft margin loss。

    一个二分类任务，计算输入 :math:`x` 和真实标签 :math:`y` （包含1或-1）之间的损失。

    .. math::
        \text{loss}(x, y) = \sum_i \frac{\log(1 + \exp(-y[i]*x[i]))}{\text{x.nelement}()}

    其中 :math:`x.nelement()` 是 :math:`x` 的元素数量。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 预测值。数据类型必须为float16、float32、bfloat16（其中，Atlas训练系列产品不支持bfloat16）。
        - **target** (Tensor) - 真实标签，shape与 `input` 相同。GE模式下类型需要与 `input` 相同。
        - **reduction** (str，可选) - 指定应用于输出结果的归约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用归约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    返回：
        Tensor或Scalar。如果 `reduction` 为 ``'none'`` ，其shape与 `input` 相同。否则，将返回Scalar。

    异常：
        - **TypeError** - 如果 `input` 或 `target` 不是Tensor。
        - **TypeError** - 如果 `input` 或 `target` 的数据类型不是float16、float32、bfloat16（其中，Atlas训练系列产品不支持bfloat16）。
        - **ValueError** - 如果 `input` 与 `target` 的shape不相同。
        - **ValueError** - 如果 `reduction` 不是 ``'none'`` 、 ``'mean'`` 或 ``'sum'`` 。
