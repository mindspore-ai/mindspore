mindspore.ops.LARSUpdate
========================

.. py:class:: mindspore.ops.LARSUpdate(epsilon=1e-05, hyperpara=0.001, use_clip=False)

    对梯度的平方和应用LARS(layer-wise adaptive rate scaling)算法。

    更多细节请参考 :class:`mindspore.nn.LARS` 。

    参数：
        - **epsilon** (float，可选) - 添加在分母中，提高数值稳定性。默认值： ``1e-05`` 。
        - **hyperpara** (float，可选) - 计算局部学习率的信任系数。默认值： ``0.001`` 。
        - **use_clip** (bool，可选) - 计算局部学习速率时是否裁剪。默认值： ``False`` 。

    输入：
        - **weight** (Tensor) - 权重Tensor，shape: :math:`(N, *)` ，其中 :math:`*` 表示任意数量的附加维度。
        - **gradient** (Tensor) - `weight` 的梯度，与 `weight` 的shape和数据类型相同。
        - **norm_weight** (Tensor) - 标量Tensor，权重的平方和。
        - **norm_gradient** (Tensor) - 标量Tensor，梯度的平方和。
        - **weight_decay** (Union[Number, Tensor]) - 衰减率。必须为标量Tensor或Number。
        - **learning_rate** (Union[Number, Tensor]) - 学习率。必须为标量Tensor或Number。

    输出：
        Tensor，计算后的梯度。

    异常：
        - **TypeError** - `epsilon` 或 `hyperpara` 不是float类型。
        - **TypeError** - `use_clip` 不是bool类型。
        - **TypeError** - `weight` 、 `gradient` 、 `norm_weight` 或 `norm_gradient` 不是Tensor。
        - **TypeError** - `weight_decay` 或 `learning_rate` 非Number或Tensor。
        - **TypeError** - `gradient` 与 `weight` 的shape不同。
