mindspore.dataset.vision.AdjustSaturation
=========================================

.. py:class:: mindspore.dataset.vision.AdjustSaturation(saturation_factor)

    调整输入图像的饱和度。

    参数：
        - **saturation_factor** (float) - 饱和度调节因子，需为非负数。输入 ``0`` 值将得到全黑图像， ``1`` 值将得到原始图像，
          ``2`` 值将调整图像饱和度为原来的2倍。

    异常：
        - **TypeError** - 如果 `saturation_factor` 不是float类型。
        - **ValueError** - 如果 `saturation_factor` 小于0。
        - **RuntimeError** - 如果输入图像的形状不是<H, W, C>。
        - **RuntimeError** - 如果输入图像的通道数不是3。

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
