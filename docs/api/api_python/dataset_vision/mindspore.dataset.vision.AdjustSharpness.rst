mindspore.dataset.vision.AdjustSharpness
========================================

.. py:class:: mindspore.dataset.vision.AdjustSharpness(sharpness_factor)

    调整输入图像的锐度。

    支持 Ascend 硬件加速，需要通过 `.device("Ascend")` 方式开启。

    参数：
        - **sharpness_factor** (float) - 锐度调节因子，需为非负数。输入 ``0`` 值将得到模糊图像， ``1`` 值将得到原始图像，
          ``2`` 值将调整图像锐度为原来的2倍。

    异常：
        - **TypeError** - 如果 `sharpness_factor` 不是float类型。
        - **ValueError** - 如果 `sharpness_factor` 小于0。
        - **RuntimeError** - 如果输入图像的形状不是<H, W, C>或<H, W>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        - 当执行设备是 Ascend 时，输入数据支持 `uint8` 或者 `float32` 类型，输入数据的通道仅支持1和3。输入数据的高度限制范围为[4, 8192]、宽度限制范围为[6, 4096]。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``CPU`` 和 ``Ascend`` 。默认值： ``CPU`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为 ``CPU`` / ``Ascend`` 。
