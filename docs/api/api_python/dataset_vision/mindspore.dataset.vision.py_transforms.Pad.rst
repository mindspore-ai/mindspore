mindspore.dataset.vision.py_transforms.Pad
==========================================

.. py:class:: mindspore.dataset.vision.py_transforms.Pad(padding, fill_value=0, padding_mode=Border.CONSTANT)

    对输入PIL图像的各边进行填充。

    **参数：**

    - **padding** (Union[int, sequence]) - 图像各边填充的像素数。若输入整型，将以该值对所有边框进行填充；若输入2元素序列，将以第一个值填充左、上边框，第二个值填充右、下边框；若输入4元素序列，将分别用于填充左、上、右和下边框。
    - **fill_value** (Union[int, tuple]，可选) - 用于填充边框的像素值，仅当 `padding_mode` 为 Border.CONSTANT 时生效 。若输入整型，将以该值填充RGB通道；若输入3元素元组，将分别用于填充R、G、B通道。默认值：0。
    - **padding_mode** (Border，可选) - 填充方式，取值可为 Border.CONSTANT、Border.EDGE、Border.REFLECT 或 Border.SYMMETRIC。默认值：Border.CONSTANT。

      - **Border.CONSTANT**：使用常量值进行填充。
      - **Border.EDGE**：使用各边的边界像素值进行填充。
      - **Border.REFLECT**：以各边的边界为轴进行镜像填充，忽略边界像素值。
      - **Border.SYMMETRIC**：以各边的边界为轴进行对称填充，包括边界像素值。

    **异常：**

    - **TypeError** - 当 `padding` 的类型不为整型或整型序列。
    - **TypeError** - 当 `fill_value` 的类型不为整型或整型序列。
    - **TypeError** - 当 `padding_mode` 的类型不为 :class:`mindspore.dataset.vision.Border` 。
    - **ValueError** - 当 `padding` 为负数。
    - **ValueError** - 当 `fill_value` 取值不在[0, 255]范围内。
    - **RuntimeError** 当输入图像的shape不为<H, W>或<H, W, C>。
