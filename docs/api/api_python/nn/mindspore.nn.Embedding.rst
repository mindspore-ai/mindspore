mindspore.nn.Embedding
=======================

.. py:class:: mindspore.nn.Embedding(vocab_size, embedding_size, use_one_hot=False, embedding_table="normal", dtype=mstype.float32, padding_idx=None)

    嵌入层。

    用于存储词向量并使用索引进行检索，根据输入Tensor中的id，从 `embedding_table` 中查询对应的embedding向量。当输入为id组成的序列时，输出为对应embedding向量构成的矩阵。

    .. note:: 
        当 `use_one_hot` 等于True时，x的类型必须是mindpore.int32。

    参数：
        - **vocab_size** (int) - 词典的大小。
        - **embedding_size** (int) - 每个嵌入向量的大小。
        - **use_one_hot** (bool) - 指定是否使用one-hot形式。默认值： ``False`` 。
        - **embedding_table** (Union[Tensor, str, Initializer, numbers.Number]) - embedding_table的初始化方法。当指定为字符串，字符串取值请参见类 `mindspore.common.initializer <https://www.mindspore.cn/docs/zh-CN/master/api_python/mindspore.common.initializer.html>`_ 。默认值： ``"normal"`` 。
        - **dtype** (mindspore.dtype) - x的数据类型。默认值： ``mstype.float32`` 。
        - **padding_idx** (int, None) - 将 `padding_idx` 对应索引所输出的嵌入向量用零填充。默认值： ``None`` 。该功能已停用。

    输入：
        - **x** (Tensor) - Tensor的shape为 :math:`(\text{batch_size}, \text{x_length})` ，其元素为整型值，并且元素数目必须小于等于vocab_size，否则相应的嵌入向量将为零。该数据类型可以是int32或int64。

    输出：
        Tensor的shape :math:`(\text{batch_size}, \text{x_length}, \text{embedding_size})` 。

    异常：
        - **TypeError** - 如果 `vocab_size` 或者 `embedding_size` 不是整型值。
        - **TypeError** - 如果 `use_one_hot` 不是布尔值。
        - **ValueError** - 如果 `padding_idx` 是一个不在[0, `vocab_size` ]范围内的整数。
