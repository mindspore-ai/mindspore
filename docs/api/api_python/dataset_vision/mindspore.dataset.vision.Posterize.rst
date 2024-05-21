mindspore.dataset.vision.Posterize
==================================

.. py:class:: mindspore.dataset.vision.Posterize(bits)

    减少图像的颜色通道的比特位数，使图像变得高对比度和颜色鲜艳，类似于海报或印刷品的效果。

    支持 Ascend 硬件加速，需要通过 `.device("Ascend")` 方式开启。

    参数：
        - **bits** (int) - 每个颜色通道保留的位数，取值需在 [0, 8] 范围内。

    异常：
        - **TypeError** - 如果 `bits` 不是int类型。
        - **ValueError** - 如果 `bits` 不在 [0, 8] 范围内。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        - 当执行设备是 Ascend 时，输入数据仅支持 `uint8` 类型，输入数据的通道仅支持1和3。输入数据的高度限制范围为[4, 8192]、宽度限制范围为[6, 4096]。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``CPU`` 和 ``Ascend`` 。默认值： ``CPU`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为 ``CPU`` / ``Ascend`` 。
