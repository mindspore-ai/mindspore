mindspore.dataset.vision.RandomAffine
=====================================

.. py:class:: mindspore.dataset.vision.RandomAffine(degrees, translate=None, scale=None, shear=None, resample=Inter.NEAREST, fill_value=0)

    对输入图像应用随机仿射变换。

    参数：
        - **degrees** (Union[int, float, sequence]) - 旋转度数的范围。
          如果 `degrees` 是一个数字，它代表旋转范围是(-degrees, degrees)。
          如果 `degrees` 是一个序列，它代表旋转是 (min, max)。
        - **translate** (sequence, 可选) - 一个序列(tx_min, tx_max, ty_min, ty_max)用于表示水平(tx)方向和垂直(ty)方向的最小/最大平移范围，取值范围 [-1.0, 1.0]。默认值：None。
          水平和垂直偏移分别从以下范围中随机选择：(tx_min*width, tx_max*width) 和 (ty_min*height, ty_max*height)。
          如果 `translate` 是一个包含2个值的元组或列表，则 (translate[0], translate[1]) 表示水平(X)方向的随机平移范围。
          如果 `translate` 是一个包含4个值的元组或列表，则 (translate[0], translate[1]) 表示水平(X)方向的随机平移范围，(translate[2], translate[3])表示垂直(Y)方向的随机平移范围。
          如果为None，则不对图像进行任何平移。
        - **scale** (sequence, 可选) - 图像的比例因子的随机范围，必须为非负数，使用原始比例。默认值：None。
        - **shear** (Union[float, Sequence[float, float], Sequence[float, float, float, float]], 可选) - 图像的剪切因子的随机范围，必须为正数。默认值：None。
          如果是数字，则应用在 (-shear, +shear) 范围内平行于 X 轴的剪切。
          如果 `shear` 是一个包含2个值的元组或列表，则在 (shear[0],shear[1]) 范围内进行水平(X)方向的剪切变换。
          如果 `shear` 是一个包含4个值的元组或列表，则在 (shear[0],shear[1]) 范围内进行水平(X)方向的剪切变换，并在(shear[2], shear[3])范围内进行垂直(Y)方向的剪切变换。
          如果为None，则不应用任何剪切。
        - **resample** (:class:`mindspore.dataset.vision.Inter` , 可选) - 图像插值方式。它可以是 [Inter.BILINEAR、Inter.NEAREST、Inter.BICUBIC、Inter.AREA] 中的任何一个。默认值：Inter.NEAREST。

          - **Inter.BILINEAR**: 双线性插值。
          - **Inter.NEAREST**: 最近邻插值。
          - **Inter.BICUBIC**: 双三次插值。
          - **Inter.AREA**: 像素区域插值。

        - **fill_value** (Union[int, tuple[int]], 可选) - 用于填充输出图像中变换之外的区域。元组中必须有三个值，取值范围是[0, 255]。默认值：0。

    异常：
        - **TypeError** - 如果 `degrees` 不是int、float或sequence类型。
        - **TypeError** - 如果 `translate` 不是sequence类型。
        - **TypeError** - 如果 `scale` 不是sequence类型。
        - **TypeError** - 如果 `shear` 不是int、float或sequence类型。
        - **TypeError** - 如果 `resample` 不是 :class:`mindspore.dataset.vision.Inter` 的类型。
        - **TypeError** - 如果 `fill_value` 不是int或tuple[int]类型。
        - **ValueError** - 如果 `degrees` 为负数。
        - **ValueError** - 如果 `translate` 不在范围 [-1.0, 1.0] 内。
        - **ValueError** - 如果 `scale` 为负数。
        - **ValueError** - 如果 `shear` 不是正数。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。
