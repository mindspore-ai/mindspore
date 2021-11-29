.. py:method:: bucket_batch_by_length(column_names, bucket_boundaries, bucket_batch_sizes, element_length_function=None, pad_info=None, pad_to_bucket_boundary=False, drop_remainder=False)

    依据数据中元素长度进行分桶。每个桶将在满了的时候进行元素填充和批处理操作。

    对数据集中的每一条数据执行长度计算函数。然后，根据该条数据的长度和桶的边界将该数据归到特定的桶里面。当桶中数据条数达到指定的大小 `bucket_batch_sizes` 时，将根据 `pad_info` 对桶中元素进行填充，再进行批处理。这样每个批次都是满的，但也有特殊情况，每个桶的最后一个批次（batch）可能不满。

    **参数：**

    - **column_names** (list[str])：传递给长度计算函数的所有列名。
    - **bucket_boundaries** (list[int])：由各个桶的上边界值组成的列表，必须严格递增。如果有n个边界，则创建n+1个桶，分配后桶的边界如下：[0, bucket_boundaries[0])，[bucket_boundaries[i], bucket_boundaries[i+1])（其中，0<i<n-1），[bucket_boundaries[n-1], inf)。
    - **bucket_batch_sizes** (list[int])：由每个桶的批次大小组成的列表，必须包含 `len(bucket_boundaries)+1` 个元素。
    - **element_length_function** (Callable, optional)：输入包含M个参数的函数，其中M等于 `len(column_names)` ，并返回一个整数。如果未指定该参数，则 `len(column_names)` 必须为1，并且该列数据第一维的shape值将用作长度（默认为None）。
    - **pad_info** (dict, optional)：有关如何对指定列进行填充的字典对象。字典中键对应要填充的列名，值必须是包含2个元素的元组。元组中第一个元素对应要填充成的shape，第二个元素对应要填充的值。如果某一列未指定将要填充后的shape和填充值，则当前批次中该列上的每条数据都将填充至该批次中最长数据的长度，填充值为0。除非 `pad_to_bucket_boundary` 为True，否则 `pad_info` 中任何填充shape为None的列，其每条数据长度都将被填充为当前批处理中最数据的长度。如果不需要填充，请将 `pad_info` 设置为None（默认为None）。
    - **pad_to_bucket_boundary** (bool, optional)：如果为True，则 `pad_info` 中填充shape为None的列，其长度都会被填充至 `bucket_boundary-1` 长度。如果有任何元素落入最后一个桶中，则将报错（默认为False）。
    - **drop_remainder** (bool, optional)：如果为True，则丢弃每个桶中最后不足一个批次数据（默认为False）。

    **返回：**

    BucketBatchByLengthDataset，按长度进行分桶和批处理操作后的数据集对象。

    **样例：**
    
	>>> # 创建一个数据集对象，其中给定条数的数据会被组成一个批次数据
	>>> # 如果最后一个批次数据小于给定的批次大小（batch_size)，则丢弃这个批次
	>>> import numpy as np
	>>> def generate_2_columns(n):
	...     for i in range(n):
	...         yield (np.array([i]), np.array([j for j in range(i + 1)]))
	>>>
	>>> column_names = ["col1", "col2"]
	>>> dataset = ds.GeneratorDataset(generate_2_columns(8), column_names)
	>>> bucket_boundaries = [5, 10]
	>>> bucket_batch_sizes = [2, 1, 1]
	>>> element_length_function = (lambda col1, col2: max(len(col1), len(col2)))
	>>> # 将对列名为"col2"的列进行填充，填充后的shape为[bucket_boundaries[i]]，其中i是当前正在批处理的桶的索引
	>>> pad_info = {"col2": ([None], -1)}
	>>> pad_to_bucket_boundary = True
	>>> dataset = dataset.bucket_batch_by_length(column_names, bucket_boundaries,
	...                                          bucket_batch_sizes,
	...                                          element_length_function, pad_info,
	...                                          pad_to_bucket_boundary)