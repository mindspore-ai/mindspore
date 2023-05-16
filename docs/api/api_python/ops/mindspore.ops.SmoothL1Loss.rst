mindspore.ops.SmoothL1Loss
============================

.. py:class:: mindspore.ops.SmoothL1Loss(beta=1.0, reduction='none')

    计算平滑L1损失，该L1损失函数有稳健性。

    更多参考详见 :func:`mindspore.ops.smooth_l1_loss`。

    参数：
        - **beta** (float，可选) - 控制损失函数在L1Loss和L2Loss间变换的阈值，该值应大于零。默认值：``1.0`` 。
        - **reduction** (str，可选) - 对输出应用特定的规约方法：可选 ``"none"`` 、 ``"mean"`` 、 ``"sum"`` 。默认值：``"none"`` 。

    输入：
        - **logits** (Tensor) - shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。数据类型支持float16或float32。
        - **labels** (Tensor) - shape： :math:`(N, *)` ，与 `logits` 的shape和数据类型相同。

    输出：
        Tensor，损失值，与 `logits` 的shape和数据类型相同。
