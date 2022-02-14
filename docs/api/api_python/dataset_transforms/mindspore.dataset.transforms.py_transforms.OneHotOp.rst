mindspore.dataset.transforms.py_transforms.OneHotOp
===================================================

.. py:class:: mindspore.dataset.transforms.py_transforms.OneHotOp(num_classes, smoothing_rate=0.0)

    将Tensor进行OneHot编码，可以进一步对标签进行平滑处理。

    **参数：**

    - **num_classes** (int) - 数据集的类别数，它应该大于数据集中最大的label编号。
    - **num_classes** (float，可选) - 标签平滑的系数，默认值：0.0。

    **异常：**
      
    - **TypeError** - 参数 `num_classes` 类型不为 int。
    - **TypeError** - 参数 `smoothing_rate` 类型不为 float。
    - **ValueError** - 参数 `smoothing_rate` 取值范围不为[0.0, 1.0]。
