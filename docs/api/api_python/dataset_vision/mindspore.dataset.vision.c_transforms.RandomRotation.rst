mindspore.dataset.vision.c_transforms.RandomRotation
=================================================

.. py:class:: mindspore.dataset.vision.c_transforms.RandomRotation(degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0)

    在指定的角度范围内，随机旋转输入图像。

    **参数：**

    - **degrees** (Union[int, float, sequence]) - 旋转角度的随机选取范围。若输入单个数字，则从(-degrees, degrees)中随机生成旋转角度；若输入2元素序列，需按(min, max)顺序排列。
    - **resample** (Inter mode, optional) - 插值方式。 它可以是 [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC] 中的任何一个，默认值：Inter.NEAREST。

      - Inter.BILINEAR，双线性插值。
      - Inter.NEAREST，最近邻插值。
      - Inter.BICUBIC，双三次插值。

    - **expand** (bool, optional) - 图像大小拓宽标志，若为True，将扩展图像大小使其足以容纳整个旋转图像；若为False或未指定，则保持输出图像与输入图像大小一致。请注意，扩展时将假设图像为中心旋转且未进行平移，默认值：False。
    - **center** (tuple, optional) - 可选的旋转中心，以图像左上角为原点，旋转中心的位置按照 (width, height) 格式指定。默认值：None，表示中心旋转。
    - **fill_value** (Union[int, tuple], optional) - 旋转图像之外区域的像素填充值。若输入3元素元组，将分别用于填充R、G、B通道；若输入整型，将以该值填充RGB通道。 `fill_value` 值必须在 [0, 255] 范围内，默认值：0。
