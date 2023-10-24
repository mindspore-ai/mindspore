mindspore.dataset.vision.AdjustHue
==================================

.. py:class:: mindspore.dataset.vision.AdjustHue(hue_factor)

    调整输入图像的色调。

    参数：
        - **hue_factor** (float) - 色调通道调节值，值必须在 [-0.5, 0.5] 范围内。

    异常：
        - **TypeError** - 如果 `hue_factor` 不是float类型。
        - **ValueError** - 如果 `hue_factor` 不在[-0.5, 0.5]的范围内。
        - **RuntimeError** - 如果输入图像的形状不是<H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``CPU`` 。默认值： ``CPU`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为 ``CPU`` 。
