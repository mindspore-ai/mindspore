mindspore.dataset.vision.ToType
===============================

.. py:class:: mindspore.dataset.vision.ToType(data_type)

    将输入转换为指定的MindSpore数据类型或NumPy数据类型。

    效果同 :class:`mindspore.dataset.transforms.TypeCast` 。

    .. note:: 此操作默认通过 CPU 执行，也支持异构加速到 GPU 或 Ascend 上执行。

    参数：
        - **data_type** (Union[mindspore.dtype, numpy.dtype]) - 输出图像的数据类型，例如 ``numpy.float32`` 。

    异常：
        - **TypeError** - 当 `data_type` 的类型不为 :class:`mindspore.dtype` 或 :class:`numpy.dtype` 。

    教程样例：
        - `视觉变换样例库
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/dataset/vision_gallery.html>`_
