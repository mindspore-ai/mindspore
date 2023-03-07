mindspore.ops.dsplit
=====================

.. py:function:: mindspore.ops.dsplit(input, indices_or_sections)

    沿着第三轴将输入Tensor分割成多个子Tensor。等同于 `axis=2` 时的 `ops.tensor_split` 。

    参数：
        - **input** (Tensor) - 待分割的Tensor。
        - **indices_or_sections** (Union[int, tuple(int), list(int)]) - 参考 :func:`mindspore.ops.tensor_split`.

    返回：
        tuple[Tensor]。
