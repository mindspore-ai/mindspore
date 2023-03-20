mindspore.ops.histc
====================

.. py:function:: mindspore.ops.histc(input, bins=100, min=0., max=0.)

    计算Tensor的直方图。

    元素被分类到 `min` 和 `max` 之间的等宽箱中。
    如果 `min` 和 `max` 均为0，则使用数据的最小值和最大值。

    低于最小值和高于最大值的元素将被忽略。

    参数：
        - **bins** (int, 可选) - 直方图箱的数量，可选。默认值：100。若指定，则必须为正数。
        - **min** (int, float, 可选) - 范围下端（含）的可选浮点数。默认值：0.。
        - **max** (int, float, 可选) - 范围上限（含）的可选浮点数。默认值：0.。
        - **x** (Tensor) - 输入Tensor，类型支持： :math:`[float16, float32, int32]` 。

    返回：
        Tensor，类型为int32的1-D Tensor。

    异常：
        - **TypeError** - 如果 `x` 不是Tensor。
        - **TypeError** - 如果 `x` 的数据类型不支持。
        - **TypeError** - 如果属性 `min` 或 `max` 不是float类型或int类型。
        - **TypeError** - 如果属性 `bins` 不是整数。
        - **ValueError** - 如果属性 `min` > `max` 。
        - **ValueError** - 如果属性 `bins` <= 0。
