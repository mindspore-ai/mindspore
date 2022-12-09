mindspore.ops.Randperm
========================

.. py:class:: mindspore.ops.Randperm(max_length=1, pad=-1, dtype=mstype.int32)

    生成从0到n-1不重复的n个随机样本。如果 `max_length` > n，则末尾的 `max_length-n` 个元素使用 `pad` 填充。

    参数：    
        - **max_length** (int) - 取样数量，必须大于0。默认值：1。
        - **pad** (int) - 填充值。默认值：-1。
        - **dtype** (mindspore.dtype) - 输出的数据类型。默认值：mindspore.int32。

    输入：
        - **n** (Tensor) - shape为 :math:`(1,)` 的输入Tensor，其数据类型为int32或int64，须在[0, `max_length`]内取值。

    输出：
        - **output** (Tensor) - shape: (`max_length`,)，数据类型为 `dtype` 。

    异常：
        - **TypeError** - `max_length` 或 `pad` 不是int类型。
        - **TypeError** - `n` 不是Tensor。
        - **TypeError** - `n` 包含非int元素。
        - **TypeError** - `n` 包含负数。
        - **TypeError** - `dtype` 不被支持。
        - **ValueError** - `n` 超出 `dtype` 的有效范围。
        - **ValueError** - `n` 大于 `max_length` 。
