mindspore.all_finite
===================

.. py:function:: mindspore.all_finite(inputs)

    检查inputs是否是有效值（无溢出）。

    .. note::
        这是一个实验性接口，后续可能删除或修改。

        此接口只在整网训练情况下用于判断梯度是否溢出，返回结果在不同后端可能存在差异。

    参数：
        - **inputs** (Union(tuple(Tensor), list(Tensor))) - 可迭代的Tensor。
    返回：
        - Tensor, 布尔类型的标量Tensor。


mindspore.LossScaler
===================

.. py:class:: mindspore.LossScaler

    使用混合精度时，用于管理损失缩放系数（loss scaler）的抽象类。

    派生类需要实现该类的所有方法。训练过程中，`scale` 和 `unscale` 用于对损失值或梯度进行放大或缩小，以避免数据溢出；`adjust` 用于调整损失缩放系数 `scale_value` 的值。

    .. note::
        - 这是一个实验性接口，后续可能删除或修改。

    .. py:method:: scale(inputs)

        对inputs进行scale，`inputs *= scale_value`。

        **参数：**

        - inputs(Union(Tensor, tuple(Tensor))): 损失值或梯度。


    .. py:method:: unscale(inputs):

        对inputs进行unscale，`inputs /= scale_value`。

        **参数：**

        - inputs(Union(Tensor, tuple(Tensor))): 损失值或梯度。

    .. py:method:: adjust(grads_finite):

        根据梯度是否为有效值（无溢出）对 `scale_value` 进行调整。

        **参数：**

        - grads_finite(Tensor): bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。


mindspore.StaticLossScaler
===================

.. py:class:: mindspore.StaticLossScaler

    损失缩放系数不变的管理器。

    .. note::
        - 这是一个实验性接口，后续可能删除或修改。

    **参数：**

    - **scale_value** (Union(float, int)) - 缩放系数。

    .. py:method:: scale(inputs)

        对inputs进行scale，`inputs *= scale_value`。

        **参数：**

        - inputs(Union(Tensor, tuple(Tensor))): 损失值或梯度。


    .. py:method:: unscale(inputs):

        对inputs进行unscale，`inputs /= scale_value`。

        **参数：**

        - inputs(Union(Tensor, tuple(Tensor))): 损失值或梯度。

    .. py:method:: adjust(grads_finite):

        `scale_value` 值固定。

        **参数：**

        - grads_finite(Tensor): bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。


mindspore.DynamicLossScaler
===================

.. py:class:: mindspore.DynamicLossScaler

    动态调整损失缩放系数的管理器。

    动态损失缩放管理器在保证梯度不溢出的情况下，尝试确定最大的损失缩放值 `scale_value`。在梯度不溢出的情况下，`scale_value` 将会每间隔 `scale_window` 步被扩大 `scale_factor` 倍，若存在溢出情况，则会将 `scale_value` 缩小 `scale_factor` 倍，并重置计数器。

    .. note::
        - 这是一个实验性接口，后续可能删除或修改。

    **参数：**

    - **scale_value** (Union(float, int)) - 初始梯度放大系数。
    - **scale_factor** (int) - 放大/缩小倍数。
    - **scale_window** (int) - 无溢出时的连续正常step的最大数量。

    .. py:method:: scale(inputs)

        根据 `scale_value` 放大inputs。

        **参数：**

        - inputs(Union(Tensor, tuple(Tensor))): 损失值或梯度。


    .. py:method:: unscale(inputs):

        对inputs进行unscale，`inputs /= scale_value`。

        **参数：**

        - inputs(Union(Tensor, tuple(Tensor))): 损失值或梯度。

    .. py:method:: adjust(grads_finite):

        根据梯度是否为有效值（无溢出）对 `scale_value` 进行调整。

        **参数：**

        - grads_finite(Tensor): bool类型的标量Tensor，表示梯度是否为有效值（无溢出）。
