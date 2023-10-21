mindspore.dataset.transforms.OneHot
===================================

.. py:class:: mindspore.dataset.transforms.OneHot(num_classes, smoothing_rate=0.0)

    对输入标签进行OneHot编码。

    对于 shape 为 :math:`(*)` 的 1 维输入，将返回 shape 为 :math:`(*, num_classes)` 的输出，其中输入值对应的索引位置处的元素值为 1 ，其余
    位置值为 0 。若指定了标签平滑系数，还将进一步平滑各元素值，增强泛化能力。

    参数：
        - **num_classes** (int) - 标签类别总数。需大于输入标签值的最大值。
        - **smoothing_rate** (float，可选) - 标签平滑系数。取值需在[0.0, 1.0]之间。默认值： ``0.0`` ，不进行标签平滑。

    异常：
        - **TypeError** - 当 `num_classes` 不为int类型。
        - **TypeError** - 当 `smoothing_rate` 不为float类型。
        - **ValueError** - 当 `smoothing_rate` 的取值不在[0.0, 1.0]范围中。
        - **RuntimeError** - 输入标签不为int类型。
        - **RuntimeError** - 输入标签的维数不为1。
