mindspore.dataset.vision.AutoContrast
=====================================

.. py:class:: mindspore.dataset.vision.AutoContrast(cutoff=0.0, ignore=None)

    在输入图像上应用自动对比度。首先计算图像的直方图，将直方图中最亮像素的值映射为255，将直方图中最暗像素的值映射为0。

    支持 Ascend 硬件加速，需要通过 `.device("Ascend")` 方式开启。

    参数：
        - **cutoff** (float, 可选) - 输入图像直方图中需要剔除的最亮和最暗像素的百分比。该值必须在 [0.0, 50.0) 范围内。默认值： ``0.0`` 。
        - **ignore** (Union[int, sequence], 可选) - 要忽略的背景像素值，忽略值必须在 [0, 255] 范围内。默认值： ``None`` 。

    异常：
        - **TypeError** - 如果 `cutoff` 不是float类型。
        - **TypeError** - 如果 `ignore` 不是int或sequence类型。
        - **ValueError** - 如果 `cutoff` 不在[0, 50.0) 范围内。
        - **ValueError** - 如果 `ignore` 不在[0, 255] 范围内。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        - 当执行设备是 Ascend 时，输入数据支持 `uint8` 或者 `float32` 类型，输入数据的通道仅支持1和3。如果数据类型是float32，期望输入的值的范围为[0，1]。输入数据的高度限制范围为[4, 8192]、宽度限制范围为[6, 4096]。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``CPU`` 和 ``Ascend`` 。默认值： ``CPU`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为 ``CPU`` / ``Ascend`` 。
