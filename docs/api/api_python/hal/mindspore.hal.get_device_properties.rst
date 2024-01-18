mindspore.hal.get_device_properties
===================================

.. py:function:: mindspore.hal.get_device_properties(device_id, device_target=None)

    返回指定卡号设备的设备属性信息。

    .. note::
        - 若用户不指定 `device_target` ，将此参数设置为当前已经设置的后端类型。
        - 对于CPU后端，固定返回1。
        - 对于Ascend后端，必须等待后端初始化完成后，调用此接口才有信息返回，否则属性信息中的 `total_memory` 以及 `free_memory` 都为0。
        - `device_id` 在Ascend后端下会被忽略，只返回当前已占用的卡属性。

    参数：
        - **device_id** (int) - 要查询的设备id。
        - **device_target** (str，可选) - 默认值：None，必须是 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 的其中一个。

    返回：
        - GPU后端，返回 `cudaDeviceProp <https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp>`_ 。
        - Ascend后端，返回 `AscendDeviceProperties` :

          .. code-block::

              AscendDeviceProperties {
                  name(str),
                  total_memory(int),
                  free_memory(int)
              }

        - Ascend后端，返回None。
