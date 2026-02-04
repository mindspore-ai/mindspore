mindspore.mint.empty_like
=========================

.. py:function:: mindspore.mint.empty_like(input, *, dtype=None, device=None, pin_memory=False) -> Tensor

    创建一个未初始化的Tensor，shape和 `input` 相同，dtype由 `dtype` 决定，Tensor使用的内存由 `device` 决定，参数 `pin_memory` 为 ``True`` ，表示使用锁页内存。

    参数：
        - **input** (Tensor) - 任意维度的Tensor。

    关键字参数：
        - **dtype** (:class:`mindspore.dtype`, 可选) - 用来描述所创建的Tensor的 `dtype` 。如果为 ``None`` ，那么将会使\
          用 `input` 的dtype。默认值： ``None`` 。
        - **device** (str, 可选) - 指定Tensor使用的内存来源。PyNative模式下支持 ``"Ascend"`` 、 ``"npu"`` 、 ``"cpu"`` 和 ``"CPU"``。
          图模式O0下支持 ``"Ascend"`` 和 ``"npu"``。如果为 ``None`` ，那么将会使用 :func:`mindspore.set_device` 设置的值。默认值 ``None`` 。
        - **pin_memory** (bool, 可选) - 表示创建的Tensor是否使用锁页内存。如果为 ``True`` ，那么 `device` 应当为 ``"cpu"`` 或 ``"CPU"``。
          默认值 ``False`` 。

    返回：
        Tensor，具有与 `input` 相同的shape，但是数据内容没有初始化（可能是任意值）。

    异常：
        - **TypeError** - `input` 不是Tensor。
        - **RuntimeError** - 如果 `pin_memory` 为True时， `device` 不为 ``"cpu"`` 或 ``"CPU"``。
