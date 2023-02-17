mindspore.ops.adaptive_avg_pool1d
=================================

.. py:function:: mindspore.ops.adaptive_avg_pool1d(input, output_size)

    对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应平均池化操作。

    通常，输入的shape为 :math:`(N_{in}, C_{in}, L_{in})`，adaptive_avg_pool1d输出区域平均值在 :math:`L_{in}` 区间。
    输出的shape为 :math:`(N_{in}, C_{in}, L_{out})`，其中 :math:`L_{out}` 由 `output_size` 定义。

    .. note::

        :math:`L_{in}` 必须能被 `output_size` 整除。

    参数：
        - **input** (Tensor) - 输入shape为 :math:`(N_{in}, C_{in}, L_{in})`，数据类型为float16、float32。
        - **output_size** (int) - 大小为 :math:`L_{out}` 。

    返回：
        Tensor，数据类型与 `input` 相同。
        输出的shape为 :math:`(N_{in}, C_{in}, L_{out})` 。

    异常：
        - **TypeError** - 如果 `output_size` 不是int类型。
        - **TypeError** - 如果 `input` 的数据类型不是float16或者float32。
        - **ValueError** - 如果 `output_size` 小于1。
        - **ValueError** - 如果 `input` 的维度不等于3。
        - **ValueError** - 如果 `input` 的最后一维尺寸小于等于 `output_size` 。
        - **ValueError** - 如果 `input` 的最后一维尺寸不能被 `output_size` 整除。
