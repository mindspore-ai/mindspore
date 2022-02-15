mindspore.dataset.vision.c_transforms.RandomColorAdjust
=======================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomColorAdjust(brightness=(1, 1), contrast=(1, 1), saturation=(1, 1), hue=(0, 0))

    随机调整输入图像的亮度、对比度、饱和度和色调。

    **参数：**

    - **brightness**  (Union[float, list, tuple], 可选) - 亮度调整因子, 默认值：(1, 1)。不能为负。
      如果是浮点数，则从 [max(0, 1-brightness), 1+brightness] 范围内统一选择因子。
      如果它是一个序列，则代表是范围 [min, max], 从此范围中选择调整因子。
    - **contrast**  (Union[float, list, tuple], 可选) - 对比度调整因子, 默认值：(1, 1)。不能为负。
      如果是浮点数，则从 [max(0, 1-contrast), 1+contrast] 范围内统一选择因子。
      如果它是一个序列，则代表是范围 [min, max], 从此范围中选择调整因子。
    - **saturation**  (Union[float, list, tuple], 可选) - 饱和度调整因子, 默认值：(1, 1)。不能为负。
      如果是浮点数，则从 [max(0, 1-saturation), 1+saturation] 范围内统一选择因子。
      如果它是一个序列，则代表是范围 [min, max], 从此范围中选择调整因子。
    - **hue**  (Union[float, list, tuple], 可选) - 色调调整因子, 默认值：(0, 0)。
      如果是浮点数，则代表是范围 [-hue, hue], 从此范围中选择调整因子。注意取值应为 0 <= `hue` <= 0.5。
      如果它是一个序列，则代表是范围 [min, max], 从此范围中选择调整因子。注意取值范围是 -0.5 <= min <= max <= 0.5。

    **异常：**

    - **TypeError** - 如果 `brightness` 不是float或sequence类型或元素不为float类型。
    - **TypeError** - 如果 `contrast` 不是float或sequence类型或元素不为float类型。
    - **TypeError** - 如果 `saturation` 不是float或sequence类型或元素不为float类型。
    - **TypeError** - 如果 `hue` 不是float或sequence类型或元素不为float类型。
    - **ValueError** - 如果 `brightness` 为负数。
    - **ValueError** - 如果 `contrast` 为负数。
    - **ValueError** - 如果 `saturation` 为负数。
    - **ValueError** - 如果 `hue` 不在 [-0.5, 0.5] 范围内。
    - **RuntimeError** - 如果输入图像的shape不是 <H, W, C>。
