mindspore.dataset.vision.py_transforms.RandomColorAdjust
========================================================

.. py:class:: mindspore.dataset.vision.py_transforms.RandomColorAdjust(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0))

    随机调整输入PIL图像的亮度、对比度、饱和度和色调。

    **参数：**

    - **brightness** (Union[float, sequence]，可选) - 亮度调节系数的随机选取范围，需为非负数。若输入浮点型，将从[max(0, 1 - brightness), 1 + brightness)中随机生成亮度调节系数；若输入2元素序列，需按(min, max)顺序排列。默认值：(1, 1)。
    - **contrast** (Union[float, sequence]，可选) - 对比度调节系数的随机选取范围，需为非负数。若输入浮点型，将从[max(0, 1 - contrast), 1 + contrast)中随机生成对比度调节系数；若输入2元素序列，需按(min, max)顺序排列。默认值：(1, 1)。
    - **saturation** (Union[float, sequence]，可选) - 饱和度调节系数的随机选取范围，需为非负数。若输入浮点型，将从[max(0, 1 - saturation), 1 + saturation)中随机生成饱和度调节系数；若输入2元素序列，需按(min, max)顺序排列。默认值：(1, 1)。
    - **hue** (Union[float, sequence]，可选) - 色调调节系数的随机选取范围。若输入浮点型，取值范围为[0, 0.5]，将从[-hue, hue)中随机生成色调调节系数；若输入2元素序列，元素取值范围为[-0.5, 0.5]，且需按(min, max)顺序排列。默认值：(0, 0)。

    **异常：**

    - **TypeError** - 当 `brightness` 的类型不为浮点型或浮点型序列。
    - **TypeError** - 当 `contrast` 的类型不为浮点型或浮点型序列。
    - **TypeError** - 当 `saturation` 的类型不为浮点型或浮点型序列。
    - **TypeError** - 当 `hue` 的类型不为浮点型或浮点型序列。
    - **ValueError** - 当 `brightness` 为负数。
    - **ValueError** - 当 `contrast` 为负数。
    - **ValueError** - 当 `saturation` 为负数。
    - **ValueError** - 当 `hue` 取值不在[-0.5, 0.5]范围内。
