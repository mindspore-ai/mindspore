mindspore.ops.ApplyMomentum
============================

.. py:class:: mindspore.ops.ApplyMomentum(use_nesterov=False, use_locking=False, gradient_scale=1.0)

    使用动量算法的优化器。

    更多详细信息，请参阅论文 `On the importance of initialization and momentum in deep learning <https://dl.acm.org/doi/10.5555/3042817.3043064>`_ 。

    输入的 `variable` 、 `accumulation` 和 `gradient` 的输入遵循隐式类型转换规则，使数据类型一致。如果它们具有不同的数据类型，则低精度数据类型将转换为相对最高精度的数据类型。

    有关公式和用法的更多详细信息，请参阅 :class:`mindspore.nn.Momentum` 。

    参数：    
        - **use_locking** (bool) - 是否对参数更新加锁保护。默认值： ``False`` 。
        - **use_nesterov** (bool) - 是否使用nesterov动量。默认值： ``False`` 。
        - **gradient_scale** (float) - 梯度的缩放比例。默认值： ``1.0`` 。

    输入：
        - **variable** (Parameter) - 要更新的权重。数据类型必须为float。
        - **accumulation** (Parameter) - 按动量权重计算的累积梯度值，数据类型与 `variable` 相同。
        - **learning_rate** (Union[Number, Tensor]) - 学习率，必须是float或为float数据类型的Scalar的Tensor。
        - **gradient** (Tensor) - 梯度，数据类型与 `variable` 相同。
        - **momentum** (Union[Number, Tensor]) - 动量，必须是float或为float数据类型的Scalar的Tensor。

    输出：
        Tensor，更新后的参数。

    异常：
        - **TypeError** - 如果 `use_locking` 或 `use_nesterov` 不是bool，或 `gradient_scale` 不是float。
        - **TypeError** - 如果 `var` 、 `accum` 和 `grad` 不支持数据类型转换。
