mindspore.nn.MultiFieldEmbeddingLookup
========================================

.. py:class:: mindspore.nn.MultiFieldEmbeddingLookup(vocab_size, embedding_size, field_size, param_init='normal', target='CPU', slice_mode='batch_slice', feature_num_list=None, max_norm=None, sparse=True, operator='SUM', dtype=mstype.float32)

    根据指定的索引和字段ID，返回输入Tensor的切片。此操作支持同时使用multi hot和one hot查找嵌入。

    .. note::
        当'target'设置为 ``'CPU'`` 时，此模块将使用P.EmbeddingLookup().set_device('CPU')指定'offset = 0'的查找表。

        当'target'设置为 ``'DEVICE'`` 时，此模块将使用P.Gather()指定'axis = 0'的查找表。

        具有相同 `field_ids` 的向量将由'operator'组合，例如'SUM'、'MAX'和'MEAN'。确保填充ID的 `input_values` 为零，以便忽略它们。如果字段绝对权重之和为零，最终输出将为零。该类仅支持['table_row_slice', 'batch_slice', 'table_column_slice']。对于Ascend设备上的'MAX'操作，存在  :math:`batch\_size * (seq\_length + field\_size) < 3500` 的约束。

    参数：
        - **vocab_size** (int) - 嵌入词典的大小。
        - **embedding_size** (int) - 每个嵌入向量的大小。
        - **field_size** (int) - 最终输出的字段大小。
        - **param_init** (Union[Tensor, str, Initializer, numbers.Number]) - 嵌入Tensor的初始化方法。当指定字符串时，请参见 `Initializer` 类了解字符串的值。默认值： ``'normal'`` 。
        - **target** (str) - 指定执行操作的'target'。该值必须在[ ``'DEVICE'`` ,  ``'CPU'`` ]中。默认值： ``'CPU'`` 。
        - **slice_mode** (str) - semi_auto_parallel或auto_parallel模式下的切片方式。默认值： ``'batch_slice'`` 。

          - **batch_slice** (str) - EmbeddingLookup算子会将输入的索引张量按批次(batch)进行划分，然后查找对应的嵌入向量。适用于每个样本都有相同数量索引的情况。
          - **field_slice** (str) - EmbeddingLookup算子会将输入的索引张量按特征(field)进行划分，然后查找对应的嵌入向量。适用于每个样本索引数量可能不同但是特征维度相同的情况。
          - **table_row_slice** (str) - EmbeddingLookup算子会将输入的索引张量看作一个二维表，并按行进行划分，然后查找对应的嵌入向量。
          - **table_column_slice** (str) - EmbeddingLookup算子会将输入的索引张量看作一个二维表，并按列进行划分，然后查找对应的嵌入向量。

        - **feature_num_list** (tuple) - 字段切片模式下的伴随数组（accompaniment array）。目前该参数的功能还未实现。默认值： ``None`` 。
        - **max_norm** (Union[float, None]) - 最大剪切值。数据类型必须为float16、float32。默认值： ``None`` 。
        - **sparse** (bool) - 使用稀疏模式。当'target'设置为'CPU'时，'sparse'必须为 ``true`` 。默认值： ``True`` 。
        - **operator** (str) - 字段特征的池化方法。支持 ``'SUM'`` 、 ``'MEAN'`` 和 ``'MAX'`` 。默认值： ``'SUM'`` 。
        - **dtype** (:class:`mindspore.dtype`) - Parameters的dtype。默认值： ``mstype.float32`` 。

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
        - **ValueError** - `target` 既不是 ``'CPU'`` 也不是 ``'DEVICE'``。
        - **ValueError** - `slice_mode` 不是 ``'batch_slice'``、 ``'field_slice'``、 ``'table_row_slice'`` 或 ``'table_column_slice'`` 。
        - **ValueError** - `sparse` 为False， `target` 为 ``'CPU'`` 。
        - **ValueError** - `slice_mode` 为 ``'field_slice'``， `feature_num_list` 为None。
        - **ValueError** - `operator` 不是 ``'SUM'``、 ``'MAX'`` 或 ``'MEAN'`` 。
