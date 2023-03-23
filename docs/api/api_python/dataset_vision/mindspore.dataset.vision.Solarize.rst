mindspore.dataset.vision.Solarize
=================================

.. py:class:: mindspore.dataset.vision.Solarize(threshold)

    通过反转阈值内的所有像素值，对输入图像进行曝光。

    参数：
        - **threshold** (Union[float, Sequence[float, float]]) - 反转的像素阈值范围，应该以（min，max）的格式提供，
          其中min和max是[0，255]范围内的整数，并且min <= max。如果只提供一个值或min = max，则反转大于min（max）的所有像素值。

    异常：
        - **TypeError** - 如果 `threshold` 不为float或Sequence[float, float]类型。
        - **ValueError** - 如果 `threshold` 的取值不在[0, 255]范围内。
