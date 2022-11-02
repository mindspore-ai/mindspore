mindspore.ops.SparseApplyFtrl
=============================

.. py:class:: mindspore.ops.SparseApplyFtrl(lr, l1, l2, lr_power, use_locking=False)

    根据FTRL-proximal算法更新相关参数。

    更多详细信息请参见 :class:`mindspore.nn.FTRL` 。

    参数：
        - **lr** (float) - 学习率，必须为正值。
        - **l1** (float) - l1正则化，必须大于或等于零。
        - **l2** (float) - l2正则化，必须大于或等于零。
        - **lr_power** (float) - 在训练期间控制降低学习率，必须小于或等于零。如果lr_power为零，则使用固定学习率。
        - **use_locking** (bool, 可选) - 是否对参数更新加锁保护。默认值：False。

    输入：
        - **var** (Parameter) - 要更新的权重。数据类型必须为float16或float32。shape为 :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **accum** (Parameter) - 要更新的累数值，shape和数据类型必须与 `var` 相同。
        - **linear** (Parameter) - 要更新的线性系数，shape和数据类型必须与 `var` 相同。
        - **grad** (Tensor) - 梯度，为一个Tensor。数据类型必须与 `var` 相同，且需要满足：如果 `var.shape > 1`，则 :math:`grad.shape[1:] = var.shape[1:]` 。
        - **indices** (Tensor) - `var` 和 `accum` 第一维度的索引向量，数据类型为int32或int64，且需要保证 :math:`indices.shape[0] = grad.shape[0]` 。

    输出：
        - **var** (Tensor) - shape和数据类型与 `var` 相同。
        - **accum** (Tensor) - shape和数据类型与 `accum` 相同。
        - **linear** (Tensor) - shape和数据类型与 `linear` 相同。

    异常：
        - **TypeError** - 如果 `lr` 、 `l1` 、 `l2` 或 `lr_power` 不是float类型。
        - **TypeError** - 如果 `use_locking` 不是bool。
        - **TypeError** - 如果 `var` 、 `grad` 、`linear` 或者 `grad` 的数据类型既不是float16也不是float32。
        - **TypeError** - 如果 `indices` 不是int32也不是int64类型。
        - **RuntimeError** - 如果 `var` 、 `grad` 、`linear` 或者 `grad` 不支持数据类型转换。
