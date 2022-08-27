mindspore.ops.SparseApplyFtrlV2
================================

.. py:class:: mindspore.ops.SparseApplyFtrlV2(lr, l1, l2, l2_shrinkage, lr_power, use_locking=False)

    根据FTRL-proximal算法更新相关参数。这个类比SpaseApplyFtrl类多了一个属性 `l2_shrinkage` 。

    除 `indices` 外，所有输入都遵守隐式类型转换规则，以使数据类型一致。如果它们数据类型不相同，则低精度数据类型将转换为相对最高精度的数据类型。

    参数：
        - **lr** (float) - 学习率，必须为正值。
        - **l1** (float) - L1正则化，必须大于或等于零。
        - **l2** (float) - L2正则化，必须大于或等于零。
        - **l2_shrinkage** (float) - L2收缩正则化。
        - **lr_power** (float) - 在训练期间控制降低学习率，必须小于或等于零。如果 `lr_power` 为零，则使用固定学习率。
        - **use_locking** (bool) - 如果为True，则 `var` 和 `accum` 将受到保护，不被更新。默认值：False。

    输入：
        - **var** (Parameter) - 要更新的权重。数据类型必须为float16或float32。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **accum** (Parameter) - 要更新的累加值，shape和数据类型必须与 `var` 相同。
        - **linear** (Parameter) - 要更新的线性系数，shape和数据类型必须与 `var` 相同。
        - **grad** (Tensor) - 梯度，为一个Tensor。数据类型必须与 `var` 相同，且需要满足 :math:`grad.shape[1:] = var.shape[1:] if var.shape > 1`。
        - **indices** (Tensor) - `var` 和 `accum` 第一维度的索引向量，数据类型为int32，且需要保证 :math:`indices.shape[0] = grad.shape[0]`。

    输出：
        3个Tensor组成的tuple，更新后的参数。

        - **var** (Tensor) - Tensor，shape和数据类型与输入 `var` 相同。
        - **accum** (Tensor) - Tensor，shape和数据类型与输入 `accum` 相同。
        - **linear** (Tensor) - Tensor，shape和数据类型与输入 `linear` 相同。

    异常：
        - **TypeError** - 如果 `lr` 、 `l1` 、 `l2` 、 `lr_power` 或 `use_locking` 不是float。
        - **TypeError** - 如果 `use_locking` 不是bool。
        - **TypeError** - 如果 `var` 、 `accum` 、 `linear` 或 `grad` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `indices` 的数据类型不是int32。
        - **RuntimeError** - 如果除 `indices` 参数外，其他的所有输入不支持数据类型转换。
