mindspore.dataset.vision.c_transforms.Rescale
=================================================

.. py:class:: mindspore.dataset.vision.c_transforms.Rescale(rescale, shift)

    基于给定的缩放和平移因子调整图像的尺寸大小。输出图像的尺寸大小为：output = image * rescale + shift。

    .. note:: 此操作支持通过 Offload 在 Ascend 或 GPU 平台上运行。

    **参数：**

    - **rescale** (float) - 缩放因子。
    - **shift** (float) - 平移因子。

    **异常：**

    - **TypeError** - 当 `rescale` 的类型不为float。
    - **TypeError** - 当 `shift` 的类型不为float。
