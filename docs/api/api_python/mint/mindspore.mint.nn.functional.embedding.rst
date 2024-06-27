mindspore.mint.nn.functional.embedding
======================================

.. py:function:: mindspore.mint.nn.functional.embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2.0, scale_grad_by_freq=False)

    以 `input` 中的值作为索引，从 `weight` 中查询对应的embedding向量。

    .. warning::
        在Ascend后端， `input` 的值非法将导致不可预测的行为。

    参数：
        - **input** (Tensor) - 用于检索的索引输入。取值范围： `[0, weight.shape[0])` 。
        - **weight** (Parameter) - 用于检索的数据。必须是2D输入。
        - **padding_idx** (int, 可选) - 如果给定非 ``None`` 值，则 `weight` 中对应行在反向计算中不会更新。取值范围：`[-weight.shape[0], weight.shape[0])`。默认值 ``None`` 。
        - **max_norm** (float, 可选) - 如果给定非 ``None`` 值，则先求出 `input` 指定位置的 `weight` 的p-范数结果reslut（p的值通过 `norm_type` 指定），然后对 `result > max_norm` 位置的 `weight` 进行原地更新，
          更新公式：:math:`\frac{max\_norm}{result+1e^{-7}}` 。默认值 ``None`` 。
        - **norm_type** (float, 可选) - 指定p-范数计算中的p值。默认值 ``2.0`` 。
        - **scale_grad_by_freq** (bool, 可选) - 如果值为 ``True`` ，则反向梯度值会按照 `input` 中索引值重复的次数进行缩放。默认值 ``False`` 。

    返回：
        Tensor，数据类型与 `weight` 保持一致，shape为 :math:`(*input.shape, weight.shape[1])` 。

    异常：
        - **ValueError** - `padding_idx` 的取值不在有效范围。
        - **ValueError** - `weight.shape` 非法。
        - **TypeError** - `weight` 不是 :class:`mindspore.Parameter` 。