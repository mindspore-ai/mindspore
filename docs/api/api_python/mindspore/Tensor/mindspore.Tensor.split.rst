mindspore.Tensor.split
======================

.. py:method:: mindspore.Tensor.split(axis=0, output_num=1)

    根据指定的轴和分割数量对Tensor进行分割。

    Tensor将被分割为相同shape的子Tensor。要求 `self.shape(axis)` 可被 `output_num` 整除。

    参数：
        - **axis** (int) - 指定分割轴。默认值：0。
        - **output_num** (int) - 指定分割数量。其值为正整数。默认值：1。

    返回：
        tuple[Tensor]，每个输出Tensor的shape相同，即 :math:`(y_1, y_2, ..., y_S)` 。数据类型与Tensor相同。

    异常：
        - **TypeError** - `axis` 或 `output_num` 不是int。
        - **ValueError** - `axis` 超出[-len(`self.shape`), len(`self.shape`))范围。或 `output_num` 小于或等于0。
        - **ValueError** - `self.shape(axis)` 不可被 `output_num` 整除。