mindspore.dataset.vision.Solarize
=================================

.. py:class:: mindspore.dataset.vision.Solarize(threshold)

    通过反转阈值内的所有像素值，对输入图像进行曝光。

    参数：
        - **threshold** (Union[float, Sequence[float, float]]) - 反转的像素取值范围，取值需在[0, 255]范围内。
          如果输入float，将反转所有大于该值的像素；
          如果输入Sequence[float, float]，将分别表示反转像素范围的左右边界。

    异常：
        - **TypeError** - 如果 `threshold` 不为float或Sequence[float, float]类型。
        - **ValueError** - 如果 `threshold` 的取值不在[0, 255]范围内。
