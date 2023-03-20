mindspore.ops.ApplyFtrl
=========================

.. py:class:: mindspore.ops.ApplyFtrl(use_locking=False)

    根据FTRL算法更新相关参数。

    更多详细信息请参见 :class:`mindspore.nn.FTRL` 。

    .. note::
        当前在Ascend平台上仅支持正数，其余场景计算结果未定义。

    参数：
        - **use_locking** (bool) - 是否对参数更新加锁保护。默认值：False。

    输入：
        - **var** (Parameter) - 要更新的权重。数据类型必须为float16或float32。shape： :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **accum** (Parameter) - 要更新的累积，shape和数据类型必须与 `var` 相同。
        - **linear** (Parameter) - 要更新的线性系数，shape和数据类型必须与 `var` 相同。
        - **grad** (Tensor) - 梯度。数据类型必须为float16或float32。
        - **lr** (Union[Number, Tensor]) - 学习率，必须为正值。它必须是float或数据类型为float16或float32的Scalar的Tensor。默认值：0.001。
        - **l1** (Union[Number, Tensor]) - l1正则化，必须大于或等于零。它必须是float类型或数据类型为float16或float32的Scalar的Tensor。默认值：0.0。
        - **l2** (Union[Number, Tensor]) - l2正则化，必须大于或等于零。它必须是float类型或数据类型为float16或float32的Scalar的Tensor。默认值：0.0。
        - **lr_power** (Union[Number, Tensor]) - 在训练期间控制降低学习率，必须小于或等于零。如果lr_power为零，则使用固定学习率。它必须是float类型或数据类型为float16或float32的Scalar的Tensor。默认值：-0.5。

    输出：
        - **var** (Tensor) - 表示更新后的 `var` 。由于输入参数已更新，因此当平台为GPU时，此值始终为零。 

    异常：
        - **TypeError** - 如果 `use_locking` 不是bool。
        - **TypeError** - 如果 `var` 、 `grad` 、 `lr` 、 `l1` 、 `l2` 或 `lr_power` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `lr` 、 `l1` 、 `l2` 或 `lr_power` 既不是数值型也不是Tensor。
        - **TypeError** - 如果 `grad` 不是Tensor。
        - **RuntimeError** - 如果 `var`、 `accum`、 `linear` 参数类型不一致。
        - **RuntimeError** - 如果 `grad`、 `lr`、 `l1`、 `l2`、`lr_power` 的参数类型与 `var` 不一致且精度大于 `var`。
