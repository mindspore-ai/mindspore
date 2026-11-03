mindspore.hal.get_device_properties
===================================

.. py:function:: mindspore.hal.get_device_properties(device_id, device_target=None)

    返回指定设备的属性信息，此接口将在后续版本中废弃。

    .. note::
        - 对于Ascend设备，必须等待设备初始化完成后，调用此接口才有信息返回，否则属性信息中的 `total_memory` 以及 `free_memory` 都为0。
        - `device_id` 在Ascend设备下会被忽略，只返回当前已占用的卡属性。

    参数：
        - **device_id** (int) - 设备id。
        - **device_target** (str，可选) - 目标设备，可选值为 ``"CPU"`` ， ``"GPU"`` 以及 ``"Ascend"`` 。默认 ``None``，表示当前已经设置的设备。

    返回：
        - GPU设备，返回 `cudaDeviceProp` :

          .. code-block::

              cudaDeviceProp {
                  name(str),
                  major(int),
                  minor(int),
                  is_multi_gpu_board(int),
                  is_integrated(int),
                  multi_processor_count(int),
                  total_memory(int),
                  warp_size(int)
              }

        - Ascend设备，返回 `AscendDeviceProperties` :

          .. code-block::

              AscendDeviceProperties {
                  name(str),
                  total_memory(int),
                  free_memory(int)
              }

        - CPU设备，返回None。
