mindspore.Tensor.randperm
==========================

.. py:method:: mindspore.Tensor.randperm(self, max_length=1, pad=-1)

    生成从0到n-1的n个随机样本，不重复。如果`max_length`>n，最后的`maxlength-n`元素将填充`pad`。

    参数：
        - **max_length** (int) - 预期获取的项数，该数字必须大于0。默认值：1。
        - **pad** (int) - 要填充的pad值。默认值：-1。
        - **dtype** (mindspore.dtype) - 输出的类型。默认值：mindspore.int32。

    输出：
        Tensor，形状为：(`max_length`,)，类型为：`dtype`。

    异常：
        - **TypeError** - 如果`max_length`或`pad`不是int。
        - **TypeError** - 如果`self`有非int元素。
        - **TypeError** - 如果`self`有负数元素。

    平台：
        ``Ascend`` ``GPU``
