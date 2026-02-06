mindspore.Tensor.new_empty
==========================

.. py:method:: mindspore.Tensor.new_empty(size, *, dtype=None, device=None) -> Tensor

    创建一个数据没有初始化的Tensor。参数 `size` 指定Tensor的shape，参数 `dtype` 指定填充值的数据类型，参数 `device` 指定Tensor使用的内存来源。

    参数：
        - **size** (Union[tuple[int], list[int], int]) - 指定输出Tensor的shape，只允许正整数或者包含正整数\
          的tuple、list。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的 `dtype` 。如果为 ``None`` ，那么将会使\
          用 `self` 的dtype。默认值： ``None`` 。
        - **device** (str, 可选) - 指定Tensor使用的内存来源。PyNative模式下支持 ``"Ascend"`` 、 ``"npu"`` 、 ``"cpu"`` 和 ``"CPU"``。
          图模式O0下支持 ``"Ascend"`` 和 ``"npu"``。如果为 ``None`` ，那么将会使用 :func:`mindspore.set_device` 设置的值。默认值 ``None`` 。

    返回：
        Tensor，shape、dtype和device由输入定义，但是数据内容没有初始化（可能是任意值）。

    异常：
        - **TypeError** - 如果 `size` 不是一个int，或元素为int的元组、列表。
