mindspore.mint.empty
====================

.. py:function:: mindspore.mint.empty(*size, *, dtype=None, device=None, pin_memory=False) -> Tensor

    创建一个数据没有初始化的Tensor。参数 `size` 指定Tensor的shape，参数 `dtype` 指定填充值的数据类型，参数 `device` 指\
    定Tensor使用的内存来源，参数 `pin_memory` 为 ``True`` ，表示使用锁页内存。

    参数：
        - **size** (Union[tuple[int], list[int], int]) - 指定输出Tensor的shape，可以是可变数量的正整数或包含正整数的tuple、list。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的dtype。如果为 ``None`` ，那\
          么将会使用 `mindspore.float32` 。默认值： ``None`` 。
        - **device** (str, 可选) - 指定Tensor使用的内存来源。PyNative模式下支持 ``"Ascend"`` 、 ``"npu"`` 、 ``"cpu"`` 和 ``"CPU"``。
          图模式O0下支持 ``"Ascend"`` 和 ``"npu"``。如果为 ``None`` ，那么将会使用 :func:`mindspore.set_device` 设置的值。默认值 ``None`` 。
        - **pin_memory** (bool, 可选) - 表示创建的Tensor是否使用锁页内存。如果为 ``True`` ，那么 `device` 应当为 ``"cpu"`` 或 ``"CPU"``。
          默认值 ``False`` 。

    返回：
        Tensor，shape、dtype和device由输入定义。

    异常：
        - **TypeError** - 如果 `size` 不是一个int，或元素为int的元组、列表。
        - **RuntimeError** - 如果 `pin_memory` 为True时， `device` 不为 ``"cpu"`` 或 ``"CPU"``。
