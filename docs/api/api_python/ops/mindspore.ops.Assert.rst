mindspore.ops.Assert
=====================

.. py:class:: mindspore.ops.Assert(summarize)

    判断给定条件是否为True，如果不为True的话则以list的形式打印 `input_data` 中的Tensor，否则继续往下运行代码。

    参数：
        - **summarize** (int) - 当判断结果为False时，打印 `input_data` 中的Tensor的数目。

    输入：
        - **condition** ([Union[Tensor[bool], bool]]) - 需要进行判断的条件。
        - **input_data** (Union(tuple[Tensor], list[Tensor])) - 当 `condition` 被判断为False的时候将被打印的Tensor。

    异常：
        - **TypeError** -  `summarize` 的数据类型不是int。
        - **TypeError** -  `condition` 的数据格式不是Tensor或bool。
        - **TypeError** -  `input_data` 的数据格式不是list或tuple。