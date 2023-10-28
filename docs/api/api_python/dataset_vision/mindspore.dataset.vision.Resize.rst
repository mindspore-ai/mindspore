mindspore.dataset.vision.Resize
============================================

.. py:class:: mindspore.dataset.vision.Resize(size, interpolation=Inter.LINEAR)

    对输入图像使用给定的 :class:`mindspore.dataset.vision.Inter` 插值方式去调整为给定的尺寸大小。

    参数：
        - **size** (Union[int, Sequence[int]]) - 图像的输出尺寸大小。若输入整型，将调整图像的较短边长度为 `size` ，且保持图像的宽高比不变；若输入是2元素组成的序列，其输入格式需要是 (高度, 宽度) 。

          - **CPU模式**：通过 `.device("CPU")` 设定执行设备为 CPU 时，`size` 取值范围：[1, 16777216]。
          - **Ascend模式**：通过 `.device("Ascend")` 设定执行设备为 Ascend 时，`size` 取值范围：[6, 32768]。

        - **interpolation** (:class:`~.vision.Inter`, 可选) - 图像插值方法。可选值详见 :class:`mindspore.dataset.vision.Inter` 。
          默认值： ``Inter.LINEAR``。

          - **Ascend模式**：通过 `.device("Ascend")` 设定执行设备为 Ascend 时， `Inter.ANTIALIAS` 、 `Inter.AREA` 、 `Inter.PILCUBIC` 差值方法不支持。

    异常：
        - **TypeError** - 当 `size` 的类型不为int或Sequence[int]。
        - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **ValueError** - 当 `size` 不为正数。
        - **RuntimeError** - 如果输入的Tensor不是 <H, W> 或 <H, W, C> 格式。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        - 当执行设备是 CPU 时，输入数据支持 `uint8` 、 `float32` 或者 `float64` 类型。
        - 当执行设备是 Ascend 时，输入数据支持 `uint8` 或者 `float32` 类型。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``CPU`` 。默认值： ``CPU`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为 ``CPU`` 。
