mindspore.mint.sum
==================

.. py:function:: mindspore.mint.sum(input, dim=None, keepdim=False, *, dtype=None)

    计算Tensor指定维度元素的和。

    .. note::
        Tensor类型的 `dim` 仅用作兼容旧版本，不推荐使用。

    参数：
        - **input** (Tensor) - 输入Tensor。
        - **dim** (Union[None, int, tuple(int), list(int), Tensor]) - 求和的维度。如果 `dim` 为 ``None`` ，对Tensor中的所有元素求和。
          如果 `dim` 为int组成的tuple或list，将对tuple中的所有维度求和，取值范围必须在 :math:`[-input.ndim, input.ndim)` 。默认值： ``None`` 。
        - **keepdim** (bool) - 是否保留输出Tensor的维度，如果为 ``True`` ，保持对应的维度且长度为1。如果为 ``False`` ，不保持维度。默认值： ``False`` 。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 期望输出Tensor的类型。默认值： ``None`` 。

    返回：
        Tensor， `input` 指定维度的和。

    异常：
        - **TypeError** - `input` 不是Tensor类型。
        - **TypeError** - `dim` 类型不是int，tulpe(int)，list(int)，Tensor或None。
        - **ValueError** - `dim` 取值不在 :math:`[-input.ndim, input.ndim)` 范围。
        - **TypeError** - `keepdim` 不是bool类型。
