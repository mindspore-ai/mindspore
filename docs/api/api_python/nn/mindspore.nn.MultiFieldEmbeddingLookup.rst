mindspore.nn.MultiFieldEmbeddingLookup
========================================

.. py:class:: mindspore.nn.MultiFieldEmbeddingLookup(vocab_size, embedding_size, field_size, param_init='normal', target='CPU', slice_mode='batch_slice', feature_num_list=None, max_norm=None, sparse=True, operator='SUM')

    根据指定的索引和字段ID，返回输入Tensor的切片。此操作支持同时使用multi hot和one hot查找嵌入。

    .. note::
        当'target'设置为'CPU'时，此模块将使用P.EmbeddingLookup().add_prim_attr('primitive_target', 'CPU')指定'offset = 0'的查找表。

        当'target'设置为'DEVICE'时，此模块将使用P.Gather()指定'axis = 0'的查找表。

        具有相同 `field_ids` 的向量将由'operator'组合，例如'SUM'、'MAX'和'MEAN'。确保填充ID的 `input_values` 为零，以便忽略它们。如果字段绝对权重之和为零，最终输出将为零。该类仅支持['table_row_slice', 'batch_slice', 'table_column_slice']。对于Ascend设备上的'MAX'操作，存在  :math:`batch\_size * (seq\_length + field\_size) < 3500` 的约束。

    参数：
        - **vocab_size** (int) - 嵌入词典的大小。
        - **embedding_size** (int) - 每个嵌入向量的大小。
        - **field_size** (int) - 最终输出的字段大小。
        - **param_init** (Union[Tensor, str, Initializer, numbers.Number]) - 嵌入Tensor的初始化方法。当指定字符串时，请参见 `Initializer` 类了解字符串的值。默认值：'normal'。
        - **target** (str) - 指定执行操作的'target'。该值必须在['DEVICE', 'CPU']中。默认值：'CPU'。
        - **slice_mode** (str) - semi_auto_parallel或auto_parallel模式下的切片方式。该值必须通过 :class:`.nn.EmbeddingLookup` 获得。默认值：'nn.EmbeddingLookup.BATCH_SLICE'。
        - **feature_num_list** (tuple) - 字段切片模式下的伴随数组（accompaniment array）。目前该参数的功能还未实现。默认值：None。
        - **max_norm** (Union[float, None]) - 最大剪切值。数据类型必须为float16、float32或None。默认值：None。
        - **sparse** (bool) - 使用稀疏模式。当'target'设置为'CPU'时，'sparse'必须为true。默认值：True。
        - **operator** (str) - 字段特征的池化方法。支持'SUM'、'MEAN'和'MAX'。默认值：'SUM'。

    输入：
        - **input_indices** (Tensor) - 指定输入Tensor元素的索引，其shape为 :math:`(batch\_size, seq\_length)` 。数据类型为int32、int64。
        - **input_values** (Tensor) - 指定 `input_indices` 元素的权重。将检索出的向量乘以 `input_values` 。其shape为 :math:`(batch\_size, seq\_length)` 。类型为float32。
        - **field_ids** (Tensor) - 指定 `input_indices` 元素的字段ID，其shape为 :math:`(batch\_size, seq\_length)` 。类型为int32。

    输出：
        Tensor，shape为 :math:`(batch\_size, field\_size, embedding\_size)` 。类型为float32。

    异常：
        - **TypeError** - `vocab_size` 、 `embedding_size` 或 `field_size` 不是int。
        - **TypeError** - `sparse` 不是bool或 `feature_num_list` 不是tuple。
        - **ValueError** - `vocab_size` 、 `embedding_size` 或 `field_size` 小于1。
        - **ValueError** - `target` 既不是'CPU'也不是'DEVICE'。
        - **ValueError** - `slice_mode` 不是'batch_slice'、'field_slice'、'table_row_slice'或'table_column_slice'。
        - **ValueError** - `sparse` 为False， `target` 为'CPU'。
        - **ValueError** - `slice_mode` 为'field_slice'， `feature_num_list` 为None。
        - **ValueError** - `operator` 不是'SUM'、'MAX'或'MEAN'。
