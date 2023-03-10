mindspore.dataset.vision.Erase
==============================

.. py:class:: mindspore.dataset.vision.Erase(top, left, height, width, value=0, inplace=False)

    使用指定的值擦除输入图像。

    参数：
        - **top** (int) - 擦除区域左上角位置的纵坐标。
        - **left** (int) - 擦除区域左上角位置的横坐标。
        - **height** (int) - 擦除区域的高度。
        - **width** (int) - 擦除区域的宽度。
        - **value** (Union[int, Sequence[int, int, int]]) - 擦除区域的像素填充值。默认值：0。
          若输入int，将以该值填充RGB通道；
          若输入Sequence[int, int, int]，将分别用于填充R、G、B通道。
        - **inplace** (bool，可选) - 是否直接在原图上执行擦除。默认值：False。

    异常：
        - **TypeError** - 如果 `top` 不是int类型。
        - **ValueError** - 如果 `top` 为负数。
        - **TypeError** - 如果 `left` 不是int类型。
        - **ValueError** - 如果 `left` 为负数。
        - **TypeError** - 如果 `height` 不是int类型。
        - **ValueError** - 如果 `height` 非正数。
        - **TypeError** - 如果 `width` 不是int类型。
        - **ValueError** - 如果 `width` 非正数。
        - **TypeError** - 如果 `value` 不是int或Sequence[int, int, int]类型。
        - **ValueError** - 如果 `value` 中元素的值不在[0, 255]范围。
        - **TypeError** - 如果 `inplace` 不是bool类型。
        - **RuntimeError** - 如果输入图像的形状不是 <H, W, C>。
