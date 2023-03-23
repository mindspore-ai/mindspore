mindspore.dataset.vision.ConvertColor
=====================================

.. py:class:: mindspore.dataset.vision.ConvertColor(convert_mode)

    更改图像的色彩空间。

    参数：
        - **convert_mode**  (:class:`mindspore.dataset.vision.ConvertMode`) - 图像色彩空间转换的模式。

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
