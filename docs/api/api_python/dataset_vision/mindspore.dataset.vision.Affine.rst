mindspore.dataset.vision.Affine
===============================

.. py:class:: mindspore.dataset.vision.Affine(degrees, translate, scale, shear, resample=Inter.NEAREST, fill_value=0)

    对输入图像进行仿射变换，保持图像中心不动。

    参数：
        - **degrees** (float) - 顺时针的旋转角度，取值需为-180到180之间。
        - **translate** (Sequence[float, float]) - 水平和垂直方向上的平移长度，需为2元素序列。
        - **scale** (float) - 放缩因子，需为正数。
        - **shear** (Union[float, Sequence[float, float]]) - 裁切度数，取值需为-180到180之间。
          若输入单个数值，表示平行于X轴的裁切角度，不进行Y轴上的裁切；
          若输入序列[float, float]，分别表示平行于X轴和Y轴的裁切角度。
        - **resample** (:class:`mindspore.dataset.vision.Inter` , 可选) - 图像插值方式。默认值：Inter.NEAREST。它可以是 [Inter.BILINEAR、Inter.NEAREST、Inter.BICUBIC、Inter.AREA] 中的任何一个。

          - **Inter.BILINEAR**: 双线性插值。
          - **Inter.NEAREST**: 最近邻插值。
          - **Inter.BICUBIC**: 双三次插值。
          - **Inter.AREA**: 像素区域插值。

        - **fill_value** (Union[int, tuple[int, int, int]], 可选) - 用于填充输出图像中变换之外的区域。元组中必须有三个值，取值范围是[0, 255]。默认值：0。

    异常：
        - **TypeError** - 如果 `degrees` 不是float类型。
        - **TypeError** - 如果 `translate` 不是Sequence[float, float]类型。
        - **TypeError** - 如果 `scale` 不是float类型。
        - **ValueError** - 如果 `scale` 非正。
        - **TypeError** - 如果 `shear` 不是float或Sequence[float, float]类型。
        - **TypeError** - 如果 `resample` 不是 :class:`mindspore.dataset.vision.Inter` 的类型。
        - **TypeError** - 如果 `fill_value` 不是int或tuple[int, int, int]类型。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
