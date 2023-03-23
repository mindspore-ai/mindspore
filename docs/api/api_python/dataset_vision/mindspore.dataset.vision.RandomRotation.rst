mindspore.dataset.vision.RandomRotation
=======================================

.. py:class:: mindspore.dataset.vision.RandomRotation(degrees, resample=Inter.NEAREST, expand=False, center=None, fill_value=0)

    在指定的角度范围内，随机旋转输入图像。

    参数：
        - **degrees** (Union[int, float, sequence]) - 旋转角度的随机选取范围。若输入单个数字，则从(-degrees, degrees)中随机生成旋转角度；若输入2元素序列，需按(min, max)顺序排列。
        - **resample** (:class:`mindspore.dataset.vision.Inter` , 可选) - 插值方式。它可以是 [Inter.BILINEAR, Inter.NEAREST, Inter.BICUBIC, Inter.AREA] 中的任何一个。默认值：Inter.NEAREST。

          - Inter.BILINEAR，双线性插值。
          - Inter.NEAREST，最近邻插值。
          - Inter.BICUBIC，双三次插值。
          - Inter.AREA，区域插值。

        - **expand** (bool, 可选) - 若为True，将扩展图像尺寸大小使其足以容纳整个旋转图像；若为False，则保持图像尺寸大小不变。请注意，扩展时将假设图像为中心旋转且未进行平移。默认值：False。
        - **center** (tuple, 可选) - 可选的旋转中心，以图像左上角为原点，旋转中心的位置按照 (宽度, 高度) 格式指定。默认值：None，表示中心旋转。
        - **fill_value** (Union[int, tuple[int]], 可选) - 旋转图像之外区域的像素填充值。若输入3元素元组，将分别用于填充R、G、B通道；若输入整型，将以该值填充RGB通道。`fill_value` 值必须在 [0, 255] 范围内。默认值：0。

    异常：
        - **TypeError** - 当 `degrees` 的类型不为int、float或sequence。
        - **TypeError** - 当 `resample` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
        - **TypeError** - 当 `expand` 的类型不为bool。
        - **TypeError** - 当 `center` 的类型不为tuple。
        - **TypeError** - 当 `fill_value` 的类型不为int或tuple[int]。
        - **ValueError** - 当 `fill_value` 取值不在[0, 255]范围内。
        - **RuntimeError** - 当输入图像的shape不为<H, W>或<H, W, C>。
