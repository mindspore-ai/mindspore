mindspore.nn.CentralCrop
=========================

.. py:class:: mindspore.nn.CentralCrop(central_fraction)

    根据指定比例裁剪出图像的中心区域。

    **参数：**

    - **central_fraction** (float) - 裁剪比例，必须是float，并且在范围(0.0, 1.0]内。

    **输入：**

    - **image** (Tensor) - shape为 :math:`(C, H, W)` 的三维Tensor，或shape为 :math:`(N,C,H,W)` 的四维Tensor。

    **输出：**

    Tensor，基于输入的三维或四维的float Tensor。

    **异常：**

    - **TypeError** - `central_fraction` 不是float。
    - **ValueError** - `central_fraction` 不在范围(0.0, 1.0]内。