mindspore.dataset.vision.Erase
==============================

.. py:class:: mindspore.dataset.vision.Erase(top, left, height, width, value=0, inplace=False)

    使用指定的值擦除输入图像。

    支持 Ascend 硬件加速，需要通过 `.device("Ascend")` 方式开启。

    参数：
        - **top** (int) - 擦除区域左上角位置的纵坐标。
        - **left** (int) - 擦除区域左上角位置的横坐标。
        - **height** (int) - 擦除区域的高度。
        - **width** (int) - 擦除区域的宽度。
        - **value** (Union[float, Sequence[float, float, float]]，可选) - 擦除区域的像素填充值。默认值： ``0`` 。
          若输入float，将以该值填充RGB通道；
          若输入Sequence[float, float, float]，将分别用于填充R、G、B通道。
        - **inplace** (bool，可选) - 是否直接在原图上执行擦除。默认值： ``False`` 。

    异常：
        - **TypeError** - 如果 `top` 不是int类型。
        - **ValueError** - 如果 `top` 为负数。
        - **TypeError** - 如果 `left` 不是int类型。
        - **ValueError** - 如果 `left` 为负数。
        - **TypeError** - 如果 `height` 不是int类型。
        - **ValueError** - 如果 `height` 非正数。
        - **TypeError** - 如果 `width` 不是int类型。
        - **ValueError** - 如果 `width` 非正数。
        - **TypeError** - 如果 `value` 不是float或Sequence[float, float, float]类型。
        - **ValueError** - 如果 `value` 中元素的值不在[0, 255]范围。
        - **TypeError** - 如果 `inplace` 不是bool类型。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_

    .. py:method:: device(device_target="CPU")

        指定该变换执行的设备。

        - 当执行设备是 Ascend 时，输入数据支持 `uint8` 或者 `float32` 类型，输入数据的通道仅支持1和3。输入数据的高度限制范围为[4, 8192]、宽度限制范围为[6, 4096]。不支持 `inplace` 参数。

        参数：
            - **device_target** (str, 可选) - 算子将在指定的设备上运行。当前支持 ``CPU`` 和 ``Ascend`` 。默认值： ``CPU`` 。

        异常：
            - **TypeError** - 当 `device_target` 的类型不为str。
            - **ValueError** - 当 `device_target` 的取值不为 ``CPU`` / ``Ascend`` 。
