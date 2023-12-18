mindspore.dataset.vision.VerticalFlip
=====================================

.. py:class:: mindspore.dataset.vision.VerticalFlip

    对输入图像进行垂直翻转。

    支持 Ascend 硬件加速，需要通过 `.device("Ascend")` 方式开启。

    异常：
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <..., H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``CPU`` 和 ``Ascend`` 。默认值： ``CPU`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为 ``CPU`` / ``Ascend`` 。
