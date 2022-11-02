mindspore.dataset.transforms.OneHot
===================================

.. py:class:: mindspore.dataset.transforms.OneHot(num_classes, smoothing_rate=0.0)

    将Tensor进行OneHot编码。

    参数：
        - **num_classes** (int) - 数据集的类别数，它应该大于数据集中最大的label编号。
        - **smoothing_rate** (float，可选) - 标签平滑的系数。默认值：0.0。

    异常：
        - **TypeError** - 参数 `num_classes` 类型不为int。
        - **TypeError** - 参数 `smoothing_rate` 类型不为float。
        - **ValueError** - 参数 `smoothing_rate` 取值范围不为[0.0, 1.0]。
        - **RuntimeError** - 输入Tensor的数据类型不为int。
        - **RuntimeError** - 参数Tensor的shape不是1-D。
