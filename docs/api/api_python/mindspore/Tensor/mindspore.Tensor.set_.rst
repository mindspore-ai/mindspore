mindspore.Tensor.set\_
================================

.. py:method:: mindspore.Tensor.set_(source=None, storage_offset=0, size=None, stride=None) -> Tensor

    设置底层存储、大小和步长。如果 source 是一个张量，则 self 张量将与之共享相同的存储，并具有相同的大小和步长。对其中一个张量元素的修改将会反映在另一个张量上。

    该方法支持多种参数组合，合法调用签名如下：

    - ``set_() -> Tensor``：
      无参调用，会将当前张量设置为一个未初始化的空张量。

    - ``set_(source: Storage) -> Tensor``：
      将 `self` 张量的底层存储设置为 ``Storage`` 。

    - ``set_(source: Storage, storage_offset: int, size: tuple | list, stride: tuple | list) -> Tensor``：
      将 `self` 张量的底层存储设置为 ``Storage``，同时将 `self` 张量的大小和步长设置为参数提供的 `size` 和 `stride` 。

    - ``set_(source: Tensor) -> Tensor``：
      将 `self` 张量与 `source` 张量共享相同的底层存储，同时 `self` 张量的 `storage_offset`、`size` 和 `stride` 与 `source` 张量相同。

    - ``set_(source: Tensor, storage_offset: int, size: tuple | list, stride: tuple | list) -> Tensor``：
      将 `self` 张量与 `source` 张量共享相同的底层存储，同时将 `self` 张量的大小和步长设置为参数提供的 `size` 和 `stride` 。
    
    .. note::
        - 如果当前张量调用 `set_` 时的设备为 ``CPU`` ，后续需要在 ``Ascend`` 上使用，建议显式将张量拷贝到 Ascend 上使用。
        - 如果当前张量调用 `set_` 时的设备为 ``CPU`` ，如果设置张量的底层存储不连续，会导致后续 ``CPU`` 上的原地（in-place）修改不生效。

    参数：
        - **source** (Tensor或Storage) - 需要共享底层存储的Tensor或Storage。
        - **storage_offset** (int) - 指定当前张量相对于底层存储的偏移量。
        - **size** (tuple或list) - 指定当前张量在底层存储中的大小。
        - **stride** (tuple或list, 可选) - 指定当前张量在底层存储中的步长。默认值： ``None`` ，默认使用行连续的步长。

    异常：
        - **TypeError** - 入参类型不符合要求，或入参个数不匹配。
        - **RuntimeError** - 如果传入的 `size` 与 `self` 原本的 `size` 相同，这时设置的底层大小超过了 `source` 的底层大小。
        - **RuntimeError** - 传入的 `storage_offset` 小于0。
        - **RuntimeError** - 设置的 `size` 中存在小于0的数。
        - **RuntimeError** - 设置的 `stride` 中存在小于0的数。
        - **RuntimeError** - 设置的 `size` 和 `stride` 的元素个数不相同。
        - **RuntimeError** - 设置的 `size` 的元素个数超过8。
        - **RuntimeError** - 当 `source` 为Tensor类型，同时参数提供了 `size` 等参数， 但 `source` 张量不连续。
        - **RuntimeError** - 当 `source` 为Tensor类型，同时没有提供其他参数， `self` 张量和 `source` 的 `dtype` 不相同。
        - **RuntimeError** - 当 `source` 为Storage类型， `self` 张量和 `source` 的 `dtype` 不相同。
        - **RuntimeError** - `self` 张量和 `source` 的 `device` 不相同。
