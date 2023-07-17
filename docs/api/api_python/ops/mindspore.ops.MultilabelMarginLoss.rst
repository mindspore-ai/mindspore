mindspore.ops.MultilabelMarginLoss
==================================

.. py:class:: mindspore.ops.MultilabelMarginLoss(reduction='mean')

    创建一个损失函数，用于最小化多分类任务的合页损失。
    它以一个2D mini-batch Tensor :math:`x` 作为输入，以包含目标类索引的2D Tensor :math:`y` 作为输出。

    更多细节请参考 :func:`mindspore.ops.multilabel_margin_loss` 。

    参数：
        - **reduction** (str，可选) - 指定应用于输出结果的规约计算方式，可选 ``'none'`` 、 ``'mean'`` 、 ``'sum'`` ，默认值： ``'mean'`` 。

          - ``"none"``：不应用规约方法。
          - ``"mean"``：计算输出元素的平均值。
          - ``"sum"``：计算输出元素的总和。

    输入：
        - **x** (Tensor) - 预测值。shape为 :math:`(C)` 或 :math:`(N, C)`，其中 :math:`N`
          为批量大小，:math:`C` 为类别数。数据类型必须为：float16或float32。
        - **target** (Tensor) - 真实标签，shape与 `input` 相同，数据类型必须为int32，标签目标由-1填充。

    输出：
        - **y** (Union[Tensor, Scalar]) - MultilabelMarginLoss损失。如果 `reduction` 的值为 'none'，
          那么返回shape为 :math:`(N)` 的Tensor类型数据。否则返回一个标量。
        - **is_target** (Tensor) - 用于反向输入的Tensor，其shape与 `target` 一致，数据类型为int32。
