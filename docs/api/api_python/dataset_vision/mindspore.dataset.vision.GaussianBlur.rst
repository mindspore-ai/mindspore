mindspore.dataset.vision.GaussianBlur
=====================================

.. py:class:: mindspore.dataset.vision.GaussianBlur(kernel_size, sigma=None)

    使用指定的高斯核对输入图像进行模糊处理。

    参数：
        - **kernel_size** (Union[int, Sequence[int, int]]) - 高斯核的大小。需为正奇数。
          若输入类型为int，将同时使用该值作为高斯核的宽、高。
          若输入类型为Sequence[int, int]，将分别使用这两个元素作为高斯核的宽、高。
        - **sigma** (Union[float, Sequence[float, float]], 可选) - 高斯核的标准差。需为正数。
          若输入类型为float，将同时使用该值作为高斯核宽、高的标准差。
          若输入类型为Sequence[float, float]，将分别使用这两个元素作为高斯核宽、高的标准差。
          默认值： ``None`` ，将通过公式 :math:`((kernel\_size - 1) * 0.5 - 1) * 0.3 + 0.8` 计算得到高斯核的标准差。

    异常：
        - **TypeError** - 如果 `kernel_size` 不是int或Sequence[int]类型。
        - **TypeError** - 如果 `sigma` 不是float或Sequence[float]类型。
        - **ValueError** - 如果 `kernel_size` 不是正数和奇数。
        - **ValueError** - 如果 `sigma` 不是正数。
        - **RuntimeError** - 如果输入图像的shape不是 <H, W> 或 <H, W, C>。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
