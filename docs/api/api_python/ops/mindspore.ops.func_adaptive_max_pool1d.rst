mindspore.ops.adaptive_max_pool1d
=================================

.. py:function:: mindspore.ops.adaptive_max_pool1d(input, output_size)

    对可以看作是由一系列1D平面组成的输入Tensor，应用一维自适应最大池化操作。

    通常，输入的shape为 :math:`(N, C, L_{in})`，adaptive_max_pool1d在 :math:`L_{in}` 区间输出区域最大值。
    输出的shape为 :math:`(N, C, L_{out})`，其中 :math:`L_{out}` 由 `output_size` 定义。

    .. note::
        - :math:`L_{in}` 必须能被 `output_size` 整除。
        - Ascend平台input参数仅支持float16类型。

    .. warning::
        从2.9.0（不含）之后版本开始，该接口将新增可选参数 `return_indices` ，接口签名将变更为
        ``mindspore.ops.adaptive_max_pool1d(input, output_size, return_indices=False)`` 。
        该变更与当前版本兼容，不使用 `return_indices` 参数时行为不变。

    参数：
        - **input** (Tensor) - 输入shape为 :math:`(N, C, L_{in})`，数据类型为float16、float32。
        - **output_size** (int) - 大小为 :math:`L_{out}` 。

    返回：
        Tensor，数据类型与 `input` 相同。
        输出的shape为 :math:`(N, C, L_{out})` 。

    异常：
        - **TypeError** - 如果 `output_size` 不是int类型。
        - **TypeError** - 如果 `input` 的数据类型不是float16或者float32。
        - **ValueError** - 如果 `output_size` 小于1。
        - **ValueError** - 如果 `input` 的维度不等于3。
        - **ValueError** - 如果 `input` 的最后一维尺寸小于 `output_size` 。
        - **ValueError** - 如果 `input` 的最后一维尺寸不能被 `output_size` 整除。
