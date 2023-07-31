mindspore.ops.SmoothL1Loss
============================

.. py:class:: mindspore.ops.SmoothL1Loss(beta=1.0, reduction='none')

    计算平滑L1损失，该L1损失函数有稳健性。

    更多参考详见 :func:`mindspore.ops.smooth_l1_loss`。

    参数：
        - **beta** (float，可选) - 控制损失函数在L1Loss和L2Loss间变换的阈值，该值应大于零。默认值： ``1.0`` 。
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'none'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    输入：
        - **logits** (Tensor) - 任意维度输入Tensor。数据类型支持float16、float32或float64。
        - **labels** (Tensor) - 真实值。shape和数据类型 与 `logits` 相同。

    输出：
        Tensor，损失值，与 `logits` 的shape和数据类型相同。
