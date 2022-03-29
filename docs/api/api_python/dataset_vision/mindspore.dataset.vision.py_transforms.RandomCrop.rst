mindspore.dataset.vision.py_transforms.RandomCrop
=================================================

.. py:class:: mindspore.dataset.vision.py_transforms.RandomCrop(size, padding=None, pad_if_needed=False, fill_value=0, padding_mode=Border.CONSTANT)

    在输入PIL图像上的随机位置，裁剪指定尺寸大小的子图。

    **参数：**

    - **size** (Union[int, Sequence[int, int]]) - 裁剪子图的尺寸大小。若输入int，则以该值为边长裁剪( `size` , `size` )尺寸大小的子图；若输入Sequence[int, int]，则以2个元素分别为高和宽裁剪子图。
    - **padding** (Union[int, Sequence[int, int], Sequence[int, int, int, int]]，可选) - 图像各边填充的像素数。指定该参数后，将在随机裁剪前对图像进行填充。若输入int，将以该值对所有边框进行填充；若输入Sequence[int, int]，将以第一个值填充左/上边框，第二个值填充右/下边框；若输入Sequence[int, int, int, int]，将分别用于填充左、上、右和下边框。默认值：None，表示不进行填充。
    - **pad_if_needed** (bool，可选) - 当图像任意边小于指定裁剪尺寸大小时，是否进行填充。默认值：False，表示不进行填充。
    - **fill_value** (Union[int, tuple[int, int, int]]，可选) - 用于填充边框的像素值，仅当 `padding_mode` 为 Border.CONSTANT 时生效 。若输入int，将以该值填充RGB通道；若输入tuple[int, int, int]，将分别用于填充R、G、B通道。默认值：0。
    - **padding_mode** (Border，可选) - 填充方式，取值可为 Border.CONSTANT、Border.EDGE、Border.REFLECT 或 Border.SYMMETRIC。默认值：Border.CONSTANT。

      - **Border.CONSTANT**：使用常量值进行填充。
      - **Border.EDGE**：使用各边的边界像素值进行填充。
      - **Border.REFLECT**：以各边的边界为轴进行镜像填充，忽略边界像素值。
      - **Border.SYMMETRIC**：以各边的边界为轴进行对称填充，包括边界像素值。

    **异常：**
        
    - **TypeError** - 当 `size` 的类型不为int或Sequence[int, int]。
    - **TypeError** - 当 `padding` 的类型不为int、Sequence[int, int]或Sequence[int, int, int, int]。
    - **TypeError** - 当 `pad_if_needed` 的类型不为bool。
    - **TypeError** - 当 `fill_value` 的类型不为int或tuple[int, int, int]。
    - **TypeError** - 当 `padding_mode` 的类型不为 :class:`mindspore.dataset.vision.Border` 。
    - **ValueError** - 当 `size` 不为正数。
    - **ValueError** - 当 `padding` 为负数。
    - **ValueError** - 当 `fill_value` 取值不在[0, 255]范围内。
