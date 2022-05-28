mindspore.ops.clip_by_norm
============================

.. py:function:: mindspore.ops.clip_by_norm(x, clip_norm, axis=None)

    基于 :math:`L_2`-norm 对输入Tensor :math:`x` 进行剪裁。如果输入Tensor :math:`x` 的 :math:`L_2`-norm 小于或者等于 :math:`clip_norm` ，原样返回输入Tensor :math:`x` 。否则，按照以下公式返回剪裁后的Tensor。

    .. math::
        \text{output}(x) = \frac{\text{clip_norm} * x}{L_2-norm(x)}.

    .. note::
        :math:`L_2`-norm 是对输入Tensor计算 `L_2` 范数。

    **参数：**

    - **x** (Tensor) - 任意维度的Tensor。数据类型是 `float16` 或者 `float32` 。
    - **clip_norm** (Tensor) - 表示裁剪比率的Tensor，数值应该大于0。Shape必须支持能广播至 `x` 的shape。数据类型是 `float16` 或者 `float32` 。
    - **axis** (Union[None, int, tuple(int), list(int)]) - 执行 :math:`L_2`-norm 计算的维度。默认值： `None` ，表示所有维度。

    **返回：**

    Tensor，表示裁剪后的Tensor。其shape和数据类型和 `x` 相同。
