mindspore.nn.EmbeddingLookup
=============================

.. py:class:: mindspore.nn.EmbeddingLookup(vocab_size, embedding_size, param_init='normal', target='CPU', slice_mode='batch_slice', manual_shapes=None, max_norm=None, sparse=True, vocab_cache_size=0)

    嵌入查找层。

    与嵌入层功能相同，主要用于自动并行或半自动并行时，存在大规模嵌入层的异构并行场景。

    .. note::
        当'target'设置为'CPU'时，此模块将使用ops.EmbeddingLookup().set_device('CPU')，在lookup表指定了'offset = 0'。
        当'target'设置为'DEVICE'时，此模块将使用ops.Gather()，在lookup表指定了'axis = 0'。
        在字段切片模式下，必须指定manual_shapes。此tuple包含vocab[i]元素, vocab[i]是第i部分的行号。

    参数：
        - **vocab_size** (int) - 嵌入词典的大小。
        - **embedding_size** (int) - 每个嵌入向量的大小。
        - **param_init** (Union[Tensor, str, Initializer, numbers.Number]) - embedding_table的初始化方法。当指定为字符串，字符串取值请参见类 `Initializer` 。默认值：'normal'。
        - **target** (str) - 指定执行操作的'target'。取值范围为['DEVICE', 'CPU']。默认值：'CPU'。
        - **slice_mode** (str) - semi_auto_parallel或auto_parallel模式下的切片方式。该值必须通过 :class:`.nn.EmbeddingLookup` 获得。默认值：'nn.EmbeddingLookup.BATCH_SLICE'。
        - **manual_shapes** (tuple) - 字段切片模式下的伴随数组（accompaniment array），默认值：None。
        - **max_norm** (Union[float, None]) - 最大剪切值。数据类型必须为float16、float32或None。默认值：None。
        - **sparse** (bool) - 使用稀疏模式。当'target'设置为'CPU'时，'sparse'必须为True。默认值：True。
        - **vocab_cache_size** (int) - 嵌入字典的缓存大小。默认值：0。仅在训练模式和'DEVICE'目标中有效。相应优化器的力矩参数也将设置为缓存大小。此外需注意，它还会消耗'DEVICE'内存，因此建议合理设置参数值，避免内存不足。

    输入：
        - **input_indices** (Tensor) - shape为 :math:`(y_1, y_2, ..., y_S)` 的Tensor。指定原始Tensor元素的索引。当取值超出embedding_table的范围时，超出部分在输出中填充为0。不支持负值，如果为负值，则结果未定义。在semi auto parallel或auto parallel模式下运行时，Input_indices只能是此接口中的二维Tensor。

    输出：
        Tensor，shape为 :math:`(z_1, z_2, ..., z_N)` 的Tensor。

    异常：
        - **TypeError** - `vocab_size` 、 `embedding_size` 或 `vocab_cache_size` 不是整数。
        - **TypeError** - `sparse` 不是bool或 `manual_shapes` 不是tuple。
        - **ValueError** - `vocab_size` 或 `embedding_size` 小于1。
        - **ValueError** - `vocab_cache_size` 小于0。
        - **ValueError** - `target` 既不是'CPU'也不是'DEVICE'。
        - **ValueError** - `slice_mode` 不是'batch_slice'、'field_slice'、'table_row_slice'或'table_column_slice'。         
        - **ValueError** - `sparse` 为False且 `target` 为'CPU'。
        - **ValueError** - `slice_mode` 为'field_slice'且 `manual_shapes` 是None。
