mindspore.dataset.vision.py_transforms.RandomResizedCrop
========================================================

.. py:class:: mindspore.dataset.vision.py_transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Inter.BILINEAR, max_attempts=10)

    在输入PIL图像上的随机位置裁剪子图，并放缩到指定大小。    

    **参数：**

    - **size** (Union[int, sequence]) - 图像放缩的大小。若输入整型，则放缩至(size, size)大小；若输入2元素序列，则以2个元素分别为高和宽放缩至(height, width)大小。
    - **scale** (Union[list, tuple]，可选) - 裁剪子图的面积相对原图比例的随机选取范围，按照(min, max)顺序排列，默认值：(0.08, 1.0)。    
    - **ratio** (Union[list, tuple]，可选) - 裁剪子图的宽高比的随机选取范围，按照(min, max)顺序排列，默认值：(3./4., 4./3.)。    
    - **interpolation** (Inter，可选) - 插值方式，取值可为 Inter.NEAREST、Inter.ANTIALIAS、Inter.BILINEAR 或 Inter.BICUBIC。默认值：Inter.BILINEAR。

      - **Inter.NEAREST**：最近邻插值。
      - **Inter.ANTIALIAS**：抗锯齿插值。
      - **Inter.BILINEAR**：双线性插值。
      - **Inter.BICUBIC**：双三次插值。

    - **max_attempts** (int，可选) - 生成随机裁剪位置的最大尝试次数，超过该次数时将使用中心裁剪，默认值：10。

    **异常：**

    - **TypeError** - 当 `size` 的类型不为整型或整型序列。
    - **TypeError** - 当 `scale` 的类型不为元组。
    - **TypeError** - 当 `ratio` 的类型不为元组。
    - **TypeError** - 当 `interpolation` 的类型不为 :class:`mindspore.dataset.vision.Inter` 。
    - **TypeError** - 当 `max_attempts` 的类型不为整型。
    - **ValueError** - 当 `size` 不为正数。
    - **ValueError** - 当 `scale` 为负数。
    - **ValueError** - 当 `ratio` 为负数。
    - **ValueError** - 当 `max_attempts` 不为正数。
