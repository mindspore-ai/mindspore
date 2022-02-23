mindspore.dataset.vision.py_transforms.RandomRotation
=====================================================

.. py:class:: mindspore.dataset.vision.py_transforms.RandomRotation(degrees, resample=<Inter.NEAREST: 0>, expand=False, center=None, fill_value=0)

    将输入PIL图像旋转随机角度。

    **参数：**

    - **degrees** (Union[int, float, sequence]) - 旋转角度的随机选取范围。若输入单个数字，则从(-degrees, degrees)中随机生成旋转角度；若输入2元素序列，需按(min, max)顺序排列。
    - **resample** (Inter，可选) - 插值方式，取值可为 Inter.NEAREST、Inter.ANTIALIAS、Inter.BILINEAR 或 Inter.BICUBIC。若输入的PIL图像模式为"1"或"P"，将直接使用 Inter.NEAREST 作为插值方式。默认为Inter.NEAREST。

      - **Inter.NEAREST**：最近邻插值。
      - **Inter.ANTIALIAS**：抗锯齿插值。
      - **Inter.BILINEAR**：双线性插值。
      - **Inter.BICUBIC**：双三次插值。

    - **expand** (bool，可选) - 若为True，将扩展图像大小使其足以容纳整个旋转图像；若为False，则保持图像大小不变。请注意，扩展时将假设图像为中心旋转且未进行平移。
    - **center** (tuple，可选) - 以图像左上角为原点，旋转中心的位置，按照(width, height)顺序排列。默认值：None，表示中心旋转。
    - **fill_value** (Union[int, tuple]，可选) - 旋转图像之外区域的像素填充值。若输入整型，将以该值填充RGB通道；若输入3元素元组，将分别用于填充R、G、B通道。默认值：0。

    **异常：**

    - **TypeError** - 当 `degrees` 的类型不为整型、浮点型或序列。
    - **TypeError** - 当 `resample` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
    - **TypeError** - 当 `expand` 的类型不为布尔型。
    - **TypeError** - 当 `center` 的类型不为元组。
    - **TypeError** - 当 `fill_value` 的类型不为整型或整型序列。
    - **ValueError** - 当 `fill_value` 取值不在[0, 255]范围内。
    - **RuntimeError** - 当输入图像的shape不为<H, W>或<H, W, C>。
