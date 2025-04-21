mindspore.dataset.vision.ConvertColor
=====================================

.. py:class:: mindspore.dataset.vision.ConvertColor(convert_mode)

    更改图像的色彩空间。

    支持 Ascend 硬件加速，需要通过 `.device("Ascend")` 方式开启。

    参数：
        - **convert_mode**  (:class:`~.vision.ConvertMode`) - 图像色彩空间转换的模式。

          - **ConvertMode.COLOR_BGR2BGRA**: 将 BGR 图像转换为 BGRA 图像。
          - **ConvertMode.COLOR_RGB2RGBA**: 将 RGB 图像转换为 RGBA 图像。
          - **ConvertMode.COLOR_BGRA2BGR**: 将 BGRA 图像转换为 BGR 图像。
          - **ConvertMode.COLOR_RGBA2RGB**: 将 RGBA 图像转换为 RGB 图像。
          - **ConvertMode.COLOR_BGR2RGBA**: 将 BGR 图像转换为 RGBA 图像。
          - **ConvertMode.COLOR_RGB2BGRA**: 将 RGB 图像转换为 BGRA 图像。
          - **ConvertMode.COLOR_RGBA2BGR**: 将 RGBA 图像转换为 BGR 图像。
          - **ConvertMode.COLOR_BGRA2RGB**: 将 BGRA 图像转换为 RGB 图像。
          - **ConvertMode.COLOR_BGR2RGB**: 将 BGR 图像转换为 RGB 图像。
          - **ConvertMode.COLOR_RGB2BGR**: 将 RGB 图像转换为 BGR 图像。
          - **ConvertMode.COLOR_BGRA2RGBA**: 将 BGRA 图像转换为 RGBA 图像。
          - **ConvertMode.COLOR_RGBA2BGRA**: 将 RGBA 图像转换为 BGRA 图像。
          - **ConvertMode.COLOR_BGR2GRAY**: 将 BGR 图像转换为 GRAY 图像。
          - **ConvertMode.COLOR_RGB2GRAY**: 将 RGB 图像转换为 GRAY 图像。
          - **ConvertMode.COLOR_GRAY2BGR**: 将 GRAY 图像转换为 BGR 图像。
          - **ConvertMode.COLOR_GRAY2RGB**: 将 GRAY 图像转换为 RGB 图像。
          - **ConvertMode.COLOR_GRAY2BGRA**: 将 GRAY 图像转换为 BGRA 图像。
          - **ConvertMode.COLOR_GRAY2RGBA**: 将 GRAY 图像转换为 RGBA 图像。
          - **ConvertMode.COLOR_BGRA2GRAY**: 将 BGRA 图像转换为 GRAY 图像。
          - **ConvertMode.COLOR_RGBA2GRAY**: 将 RGBA 图像转换为 GRAY 图像。

    异常：
        - **TypeError** - 如果 `convert_mode` 不是类 :class:`mindspore.dataset.vision.ConvertMode` 的类型。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        - 当执行设备是 Ascend 时，输入数据支持 `uint8` 或者 `float32` 类型，数据格式支持NHWC，Channels: [1, 3, 4], N只支持1。输入数据的高度限制范围为[4, 8192]、宽度限制范围为[6, 4096]。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``CPU`` 和 ``Ascend`` 。默认值： ``CPU`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为 ``CPU`` / ``Ascend`` 。
