    .. py:method:: apply(apply_func)

        对数据集对象执行给定操作函数。

        **参数：**

        `apply_func` (function)：传入 `Dataset` 对象作为参数，并将返回处理后的 `Dataset` 对象。

        **返回：**

        执行了给定操作函数的数据集对象。

        **样例：**

        >>> # dataset是数据集类的实例化对象
        >>>
        >>> # 声明一个名为apply_func函数，其返回值是一个Dataset对象
        >>> def apply_func(data)：
        ...     data = data.batch(2)
        ...     return data
        >>>
        >>> # 通过apply操作调用apply_func函数
        >>> dataset = dataset.apply(apply_func)

        **异常：**

        - **TypeError：** `apply_func` 不是一个函数。
        - **TypeError：** `apply_func` 未返回Dataset对象。

    .. py:method:: batch(batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None, input_columns=None, output_columns=None, column_order=None, pad_info=None, python_multiprocessing=False)

        将dataset中连续 `batch_size` 行数据合并为一个批处理数据。

        对一个批处理数据执行给定操作与对条数据进行给定操作用法一致。对于任意列，batch操作要求该列中的各条数据shape必须相同。如果给定可执行函数 `per_batch_map` ，它将作用于批处理后的数据。

        .. note::
            执行 `repeat` 和 `batch` 操作的顺序，会影响数据批次的数量及 `per_batch_map` 操作。建议在batch操作完成后执行repeat操作。

        **参数：**

        - **batch_size** (int or function) - 每个批处理数据包含的条数。参数需要是int或可调用对象，该对象接收1个参数，即BatchInfo。
        - **drop_remainder** (bool, optional) - 是否删除最后一个数据条数小于批处理大小的batch（默认值为False）。如果为True，并且最后一个批次中数据行数少于 `batch_size`，则这些数据将被丢弃，不会传递给后续的操作。
        - **num_parallel_workers** (int, optional) - 用于进行batch操作的的线程数（threads），默认值为None。
        - **per_batch_map** (callable, optional) - 是一个以(list[Tensor], list[Tensor], ..., BatchInfo)作为输入参数的可调用对象。每个list[Tensor]代表给定列上的一批Tensor。入参中list[Tensor]的个数应与 `input_columns` 中传入列名的数量相匹配。该可调用对象的最后一个参数始终是BatchInfo对象。`per_batch_map` 应返回(list[Tensor], list[Tensor], ...)。其出中list[Tensor]的个数应与输入相同。如果输出列数与输入列数不一致，则需要指定 `output_columns`。        - **input_columns** (Union[str, list[str]], optional)：由输入列名组成的列表。如果 `per_batch_map` 不为None，列表中列名的个数应与 `per_batch_map` 中包含的列数匹配（默认为None）。
        - **output_columns** (Union[str, list[str]], optional) - 当前操作所有输出列的列名列表。如果len(input_columns) != len(output_columns)，则此参数必须指定。此列表中列名的数量必须与给定操作的输出列数相匹配（默认为None，输出列将与输入列具有相同的名称）。
        - **column_order** (Union[str, list[str]], optional) - 指定整个数据集对象中包含的所有列名的顺序。如果len(input_column) != len(output_column)，则此参数必须指定。 注意：这里的列名不仅仅是在 `input_columns` 和 `output_columns` 中指定的列。
        - **pad_info** (dict, optional) - 用于对给定列进行填充。例如 `pad_info={"col1":([224,224],0)}` ，则将列名为"col1"的列填充到大小为[224,224]的张量，并用0填充缺失的值（默认为None)。
        - **python_multiprocessing** (bool, optional) - 针对 `per_batch_map` 函数，使用Python多进执行的方式进行调用。如果函数计算量大，开启这个选项可能会很有帮助（默认值为False）。

        **返回：**

        批处理后的数据集对象。

        **样例：**

        >>> # 创建一个数据集对象，每100条数据合并成一个批次
        >>> # 如果最后一个批次数据小于给定的批次大小（batch_size)，则丢弃这个批次
        >>> dataset = dataset.batch(100, True)
        >>> # 根据批次编号调整图像大小，如果是第5批，则图像大小调整为(5^2, 5^2) = (25, 25)
        >>> def np_resize(col, batchInfo):
        ...     output = col.copy()
        ...     s = (batchInfo.get_batch_num() + 1) ** 2
        ...     index = 0
        ...     for c in col:
        ...         img = Image.fromarray(c.astype('uint8')).convert('RGB')
        ...         img = img.resize((s, s), Image.ANTIALIAS)
        ...         output[index] = np.array(img)
        ...         index += 1
        ...     return (output,)
        >>> dataset = dataset.batch(batch_size=8, input_columns=["image"], per_batch_map=np_resize)

    .. py:method:: bucket_batch_by_length(column_names, bucket_boundaries, bucket_batch_sizes, element_length_function=None, pad_info=None, pad_to_bucket_boundary=False, drop_remainder=False)

        依据数据中元素长度进行分桶。每个桶将在满了的时候进行元素填充和批处理操作。

        对数据集中的每一条数据执行长度计算函数。然后，根据该条数据的长度和桶的边界将该数据归到特定的桶里面。当桶中数据条数达到指定的大小 `bucket_batch_sizes` 时，将根据 `pad_info` 对桶中元素进行填充，再进行批处理。这样每个批次都是满的，但也有特殊情况，每个桶的最后一个批次（batch）可能不满。

        **参数：**

        - **column_names** (list[str]) - 传递给长度计算函数的所有列名。
        - **bucket_boundaries** (list[int]) - 由各个桶的上边界值组成的列表，必须严格递增。如果有n个边界，则创建n+1个桶，分配后桶的边界如下：[0, bucket_boundaries[0])，[bucket_boundaries[i], bucket_boundaries[i+1])（其中，0<i<n-1），[bucket_boundaries[n-1], inf)。
        - **bucket_batch_sizes** (list[int]) - 由每个桶的批次大小组成的列表，必须包含 `len(bucket_boundaries)+1` 个元素。
        - **element_length_function** (Callable, optional) - 输入包含M个参数的函数，其中M等于 `len(column_names)` ，并返回一个整数。如果未指定该参数，则 `len(column_names)` 必须为1，并且该列数据第一维的shape值将用作长度（默认为None）。
        - **pad_info** (dict, optional) - 有关如何对指定列进行填充的字典对象。字典中键对应要填充的列名，值必须是包含2个元素的元组。元组中第一个元素对应要填充成的shape，第二个元素对应要填充的值。如果某一列未指定将要填充后的shape和填充值，则当前批次中该列上的每条数据都将填充至该批次中最长数据的长度，填充值为0。除非 `pad_to_bucket_boundary` 为True，否则 `pad_info` 中任何填充shape为None的列，其每条数据长度都将被填充为当前批处理中最数据的长度。如果不需要填充，请将 `pad_info` 设置为None（默认为None）。
        - **pad_to_bucket_boundary** (bool, optional) - 如果为True，则 `pad_info` 中填充shape为None的列，其长度都会被填充至 `bucket_boundary-1` 长度。如果有任何元素落入最后一个桶中，则将报错（默认为False）。
        - **drop_remainder** (bool, optional) - 如果为True，则丢弃每个桶中最后不足一个批次数据（默认为False）。

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

    .. py:method:: build_sentencepiece_vocab(columns, vocab_size, character_coverage, model_type, params)

        用于从源数据集对象创建句子词表的函数。

        **参数：**

        - **columns** (list[str]) - 指定从哪一列中获取单词。
        - **vocab_size** (int) - 词汇表大小。
        - **character_coverage** (int) - 模型涵盖的字符百分比，必须介于0.98和1.0之间。默认值如0.9995，适用于具有丰富字符集的语言，如日语或中文字符集；1.0适用于其他字符集较小的语言，比如英语或拉丁文。
        - **model_type** (SentencePieceModel) - 模型类型，枚举值包括unigram（默认值）、bpe、char及word。当类型为word时，输入句子必须预先标记。
        - **params** (dict) - 依据原始数据内容构建祠表的附加参数，无附加参数时取值可以是空字典。

        **返回：**

        SentencePieceVocab，从数据集构建的词汇表。

        **样例：**

        >>> from mindspore.dataset.text import SentencePieceModel
        >>>
        >>> # DE_C_INTER_SENTENCEPIECE_MODE 是一个映射字典
        >>> from mindspore.dataset.text.utils import DE_C_INTER_SENTENCEPIECE_MODE
        >>> dataset = ds.TextFileDataset("/path/to/sentence/piece/vocab/file", shuffle=False)
        >>> dataset = dataset.build_sentencepiece_vocab(["text"], 5000, 0.9995,
        ...                                             DE_C_INTER_SENTENCEPIECE_MODE[SentencePieceModel.UNIGRAM],
        ...                                             {})

    .. py:method:: build_vocab(columns, freq_range, top_k, special_tokens, special_first)

        基于数据集对象创建词汇表。

        用于收集数据集中所有的唯一单词，并返回 `top_k` 个最常见的单词组成的词汇表（如果指定了 `top_k` ）。

        **参数：**

        - **columns** (Union[str, list[str]]) ：指定从数据集对象中哪一列中获取单词。
        - **freq_range** (tuple[int]) - 由(min_frequency, max_frequency)组成的整数元组，在这个频率范围的词汇会被保存下来。
          取值范围需满足：0 <= min_frequency <= max_frequency <= total_words，其中min_frequency、max_frequency的默认值分别设置为0、total_words。
        - **top_k** (int) - 词汇表中包含的单词数，取 `top_k` 个最常见的单词。`top_k` 优先级低于 `freq_range`。如果 `top_k` 的值大于单词总数，则取所有单词。
        - **special_tokens** (list[str]) - 字符串列表，每个字符串都是一个特殊的标记。
        - **special_first** (bool) - 是否将 `special_tokens` 添加到词汇表首尾。如果指定了 `special_tokens` 且
          `special_first` 设置为默认值，则将 `special_tokens` 添加到词汇表最前面。

        **返回：**

        从数据集对象中构建出的词汇表对象。

        **样例：**

        >>> def gen_corpus():
        ...     # 键：单词，值：出现次数，键的取值采用字母表示有利于排序和显示。
        ...     corpus = {"Z": 4, "Y": 4, "X": 4, "W": 3, "U": 3, "V": 2, "T": 1}
        ...     for k, v in corpus.items():
        ...         yield (np.array([k] * v, dtype='S'),)
        >>> column_names = ["column1", "column2", "column3"]
        >>> dataset = ds.GeneratorDataset(gen_corpus, column_names)
        >>> dataset = dataset.build_vocab(columns=["column3", "column1", "column2"],
        ...                               freq_range=(1, 10), top_k=5,
        ...                               special_tokens=["<pad>", "<unk>"],
        ...                               special_first=True,vocab='vocab')

    .. py:method:: close_pool()

        关闭数据集对象中的多进程池。如果您熟悉多进程库，可以将此视为进程池对象的析构函数。

    .. py:method:: concat(datasets)

        对传入的多个数据集对象进行拼接操作。重载“+”运算符来进行数据集对象拼接操作。

        .. note::用于拼接的多个数据集对象，其列名、每列数据的维度（rank)和类型必须相同。

        **参数：**

        - **datasets** (Union[list, class Dataset]) - 与当前数据集对象拼接的数据集对象列表或单个数据集对象。


        **返回：**

        ConcatDataset，拼接后的数据集对象。

        **样例：**

        >>> # 通过使用“+”运算符拼接dataset_1和dataset_2，获得拼接后的数据集对象
        >>> dataset = dataset_1 + dataset_2
        >>> # 通过concat操作拼接dataset_1和dataset_2，获得拼接后的数据集对象
        >>> dataset = dataset_1.concat(dataset_2)

    .. py:method:: create_dict_iterator(num_epochs=-1, output_numpy=False)

        基于数据集对象创建迭代器，输出数据为字典类型。

        字典中列的顺序可能与数据集对象中原始顺序不同。

        **参数：**

        - **num_epochs** (int, optional) - 迭代器可以迭代的最多轮次数（默认为-1，迭代器可以迭代无限次）。
        - **output_numpy** (bool, optional) - 是否输出NumPy数据类型，如果 `output_numpy` 为False，迭代器输出的每列数据类型为MindSpore.Tensor（默认为False）。

        **返回：**

        DictIterator，基于数据集对象创建的字典迭代器。

        **样例：**

        >>> # dataset是数据集类的实例化对象
        >>> iterator = dataset.create_dict_iterator()
        >>> for item in iterator:
        ...     # item 是一个dict
        ...     print(type(item))
        ...     break
        <class 'dict'>

    .. py:method:: create_tuple_iterator(columns=None, num_epochs=-1, output_numpy=False, do_copy=True)

        基于数据集对象创建迭代器，输出数据为ndarray组成的列表。

        可以使用columns指定输出的所有列名及列的顺序。如果columns未指定，列的顺序将保持不变。

        **参数：**

        - **columns** (list[str], optional) - 用于指定列顺序的列名列表（默认为None，表示所有列）。
        - **num_epochs** (int, optional) - 迭代器可以迭代的最多轮次数（默认为-1，迭代器可以迭代无限次）。
        - **output_numpy** (bool, optional) - 是否输出NumPy数据类型，如果output_numpy为False，迭代器输出的每列数据类型为MindSpore.Tensor（默认为False）。
        - **do_copy** (bool, optional) - 当输出数据类型为mindspore.Tensor时，通过此参数指定转换方法，采用False主要考虑以获得更好的性能（默认为True）。

        **返回：**

        TupleIterator，基于数据集对象创建的元组迭代器。

        **样例：**

        >>> # dataset是数据集类的实例化对象
        >>> iterator = dataset.create_tuple_iterator()
        >>> for item in iterator：
        ...     # item 是一个列表
        ...     print(type(item))
        ...     break
        <class 'list'>

    .. py:method:: device_que(send_epoch_end=True, create_data_info_queue=False)

        返回一个能将数据传输到设备上的数据集对象。

        **参数：**

        - **send_epoch_end** (bool, optional) - 数据发送完成后是否发送结束标识到设备上（默认值为True）。
        - **create_data_info_queue** (bool, optional) - 是否创建一个队列，用于存储每条数据的type和shape（默认值为False）。


        .. note::
            如果设备类型为Ascend，数据的每一列将被依次单独传输，每次传输的数据大小限制为256M。


        **返回：**

        TransferDataset，用于帮助发送数据到设备上的数据集对象。


    .. py:method:: dynamic_min_max_shapes()

        获取数据集对象中单条数据的最小和最大shape，用于图编译过程。

        **返回：**

        列表，原始数据集对象中单条数据的最小和最大shape分别以list形式返回。

        **样例：**

        >>> import numpy as np
        >>>
        >>> def generator1():
        >>>     for i in range(1, 100):
        >>>         yield np.ones((16, i, 83)), np.array(i)
        >>>
        >>> dataset = ds.GeneratorDataset(generator1, ["data1", "data2"])
        >>> dataset.set_dynamic_columns(columns={"data1": [16, None, 83], "data2": []})
        >>> min_shapes, max_shapes = dataset.dynamic_min_max_shapes()


    .. py:method:: filter(predicate, input_columns=None, num_parallel_workers=None)

        通过判断条件对数据集对象中的数据进行过滤。

        .. note::
             如果 `input_columns` 未指定或为空，则将使用所有列。

        **参数：**

        - **predicate** (callable) - Python可调用对象，返回值为Bool类型。如果为False，则过滤掉该条数据。
        - **input_columns** (Union[str, list[str]], optional) - 输入列名组成的列表，当取默认值None时，`predicate` 将应用于数据集中的所有列。
        - **num_parallel_workers** (int, optional) - 用于并行处理数据集的线程数（默认为None，将使用配置文件中的值）。

        **返回：**

        FilterDataset，执行给定筛选过滤操作的数据集对象。

        **样例：**

        >>> # 生成一个list，其取值范围为（0，63）
        >>> # 过滤掉数值大于或等于11的数据
        >>> dataset = dataset.filter(predicate=lambda data: data < 11, input_columns = ["data"])


    .. py:method:: flat_map(func)

        对数据集对象中每一条数据执行给定的 `func` 操作，并将结果展平。

        指定的 `func` 是一个函数，输入必须为一个'ndarray'，返回值是一个'Dataset'对象。

        **参数：**

        - **func** (function) - 输入'ndarray'并返回一个'Dataset'对象的函数。

        **返回：**

        执行给定操作的数据集对象。

        **样例：**

        >>> # 以NumpySlicesDataset为例
        >>> dataset = ds.NumpySlicesDataset([[0, 1], [2, 3]])
        >>>
        >>> def flat_map_func(array):
        ...     # 使用数组创建NumpySlicesDataset
        ...     dataset = ds.NumpySlicesDataset(array)
        ...     # 将数据集对象中的数据重复两次
        ...     dataset = dataset.repeat(2)
        ...     return dataset
        >>>
        >>> dataset = dataset.flat_map(flat_map_func)
            >>> # [[0, 1], [0, 1], [2, 3], [2, 3]]

        **异常：**

        - **TypeError** - `func` 不是函数。
        - **TypeError** - `func` 的返回值不是数据集对象。

    .. py:method:: get_batch_size()

        获得批处理的大小，即一个批次中包含的数据条数。

        **返回：**

        int，一个批次中包含的数据条数。

        **样例：**

        >> # dataset是数据集类的实例化对象
        >> batch_size = dataset.get_batch_size()

    .. py:method:: get_class_indexing()

        返回类别索引。

        **返回：**

        dict，描述类别名称到索引的键值对映射关系，通常为str-to-int格式。针对COCO数据集，类别名称到索引映射关系描述形式为str-to-list<int>格式，列表中的第二个数字表示超级类别。

        **样例：**

        >> # dataset是数据集类的实例化对象
        >> class_indexing = dataset.get_class_indexing()


    .. py:method:: get_col_names()

        返回数据集对象中包含的列名。

        **返回：**

        list，数据集中所有列名组成列表。

        **样例：**

        >> # dataset是数据集类的实例化对象
        >> col_names = dataset.get_col_names()

    .. py:method:: get_dataset_size()

        返回一个epoch中的batch数。

        **返回：**

        int，batch的数目。

    .. py:method:: get_repeat_count()

        获取 `RepeatDataset` 中的repeat次数（默认为1）。

        **返回：**

        int，repeat次数。

    .. py:method:: input_indexs
        :property:

        获取input index信息。

        **返回：**

        input index信息的元组。

        **样例：**

        >>> # dataset是Dataset对象的实例
        >>> # 设置input_indexs
        >>> dataset.input_indexs = 10
        >>> print(dataset.input_indexs)
        10

    .. py:method:: map(operations, input_columns=None, output_columns=None, column_order=None, num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None)

        将operations列表中的每个operation作用于数据集。

        作用的顺序由每个operation在operations参数中的位置决定。
        将首先作用operation[0]，然后operation[1]，operation[2]，以此类推。

        每个operation将数据集中的一列或多列作为输入，并将输出零列或多列。
        第一个operation将 `input_columns` 中指定的列作为输入。
        如果operations列表中存在多个operation，则上一个operation的输出列将用作下一个operation的输入列。

        最后一个operation输出列的列名由 `output_columns` 指定。

        只有在 `column_order` 中指定的列才会传播到子节点，并且列的顺序将与 `column_order` 中指定的顺序相同。

        **参数：**

        - **operations** (Union[list[TensorOp], list[functions]]) - 要作用于数据集的operations列表。将按operations列表中显示的顺序作用在数据集。
        - **input_columns** (Union[str, list[str]], optional) - 第一个operation输入的列名列表。此列表的大小必须与第一个operation预期的输入列数相匹配。（默认为None，从第一列开始，无论多少列，都将传递给第一个operation）。
        - **output_columns** (Union[str, list[str]], optional) - 最后一个operation输出的列名列表。如果 `input_columns` 长度不等于 `output_columns` 长度，则此参数必选。此列表的大小必须与最后一个operation的输出列数相匹配（默认为None，输出列将与输入列具有相同的名称，例如，替换一些列）。
        - **column_order** (list[str], optional) - 指定整个数据集中所需的所有列的列表。当 `input_columns` 长度不等于 `output_columns` 长度时，则此参数必选。注意：这里的列表不仅仅是参数 `input_columns` 和 `output_columns` 中指定的列。
        - **num_parallel_workers** (int, optional) - 用于并行处理数据集的线程数（默认为None，将使用配置文件中的值）。
        - **python_multiprocessing** (bool, optional) - 将Python operations委托给多个工作进程进行并行处理。如果Python operations计算量很大，此选项可能会很有用（默认值为False）。
        - **cache** (DatasetCache, optional) - 使用Tensor缓存服务加快数据集处理速度（默认为None，即不使用缓存）。
        - **callbacks** (DSCallback, list[DSCallback], optional) - 要调用的Dataset回调函数列表（默认为None）。

        .. note::
            - `operations` 参数主要接收 `mindspore.dataset` 模块中c_transforms、py_transforms算子，以及用户定义的Python函数(PyFuncs)。
            - 不要将 `mindspore.nn` 和 `mindspore.ops` 或其他的网络计算算子添加到 `operations` 中。

        **返回：**

        MapDataset，map操作后的数据集。

        **样例：**

        >>> # dataset是Dataset的一个实例，它有2列，"image"和"label"。
        >>>
        >>> # 定义两个operation，每个operation接受1列输入，输出1列。
        >>> decode_op = c_vision.Decode(rgb=True)
        >>> random_jitter_op = c_vision.RandomColorAdjust(brightness=(0.8, 0.8), contrast=(1, 1),
        ...                                               saturation=(1, 1), hue=(0, 0))
        >>>
        >>> # 1）简单的map示例。
        >>>
        >>> # 在列“image"上应用decode_op。此列将被
        >>> # decode_op的输出列替换。由于未指定column_order，因此两列“image"
        >>> # 和“label"将按其原始顺序传播到下一个节点。
        >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"])
        >>>
        >>> # 解码列“image"并将其重命名为“decoded_image"。
        >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"], output_columns=["decoded_image"])
        >>>
        >>> # 指定输出列的顺序。
        >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
        ...                       output_columns=None, column_order=["label", "image"])
        >>>
        >>> # 将列“image"重命名为“decoded_image"，并指定输出列的顺序。
        >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
        ...                       output_columns=["decoded_image"], column_order=["label", "decoded_image"])
        >>>
        >>> # 将列“image"重命名为“decoded_image"，并只保留此列。
        >>> dataset = dataset.map(operations=[decode_op], input_columns=["image"],
        ...                       output_columns=["decoded_image"], column_order=["decoded_image"])
        >>>
        >>> # 使用用户自定义Python函数的map简单示例。列重命名和指定列顺序
        >>> # 的方式同前面的示例相同。
        >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
        >>> dataset = dataset.map(operations=[(lambda x: x + 1)], input_columns=["data"])
        >>>
        >>> # 2）多个operation的map示例。
        >>>
        >>> # 创建一个数据集，图像被解码，并随机颜色抖动。
        >>> # decode_op以列“image"作为输入，并输出一列。将
        >>> # decode_op输出的列作为输入传递给random_jitter_op。
        >>> # random_jitter_op将输出一列。列“image"将替换为
        >>> # random_jitter_op（最后一个operation）输出的列。所有其他
        >>> # 列保持不变。由于未指定column_order，因此
        >>> # 列的顺序将保持不变。
        >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"])
        >>>
        >>> # 将random_jitter_op输出的列重命名为“image_mapped"。
        >>> # 指定列顺序的方式与1中的示例相同。
        >>> dataset = dataset.map(operations=[decode_op, random_jitter_op], input_columns=["image"],
        ...                       output_columns=["image_mapped"])
        >>>
        >>> # 使用用户自定义Python函数的多个operation的map示例。列重命名和指定列顺序
        >>> # 的方式与1中的示例相同。
        >>> dataset = ds.NumpySlicesDataset(data=[[0, 1, 2]], column_names=["data"])
        >>> dataset = dataset.map(operations=[(lambda x: x * x), (lambda x: x - 1)], input_columns=["data"],
        ...                                   output_columns=["data_mapped"])
        >>>
        >>> # 3）输入列数不等于输出列数的示例。
        >>>
        >>> # operation[0] 是一个 lambda，它以 2 列作为输入并输出 3 列。
        >>> # operations[1] 是一个 lambda，它以 3 列作为输入并输出 1 列。
        >>> # operations[2] 是一个 lambda，它以 1 列作为输入并输出 4 列。
        >>> #
        >>> # 注：operation[i]的输出列数必须等于
        >>> # operation[i+1]的输入列。否则，map算子会
        >>> # 出错。
        >>> operations = [(lambda x, y: (x, x + y, x + y + 1)),
        ...               (lambda x, y, z: x * y * z),
        ...               (lambda x: (x % 2, x % 3, x % 5, x % 7))]
        >>>
        >>> # 注：由于输入列数与
        >>> # 输出列数不相同，必须指定output_columns和column_order
        >>> # 参数。否则，此map算子也会出错。
        >>>
        >>> dataset = ds.NumpySlicesDataset(data=([[0, 1, 2]], [[3, 4, 5]]), column_names=["x", "y"])
        >>>
        >>> # 按以下顺序将所有列传播到子节点：
        >>> dataset = dataset.map(operations, input_columns=["x", "y"],
        ...                       output_columns=["mod2", "mod3", "mod5", "mod7"],
        ...                       column_order=["mod2", "mod3", "mod5", "mod7"])
        >>>
        >>> # 按以下顺序将某些列传播到子节点：
        >>> dataset = dataset.map(operations, input_columns=["x", "y"],
        ...                       output_columns=["mod2", "mod3", "mod5", "mod7"],
        ...                       column_order=["mod7", "mod3", "col2"])

    .. py:method:: num_classes()

        获取数据集中的样本的class数目。

        **返回：**

        int，class数目。

    .. py:method:: output_shapes()

        获取输出数据的shape。

        **返回：**

        list，每列shape的列表。

    .. py:method:: output_types()

        获取输出数据类型。

        **返回：**

        list，每列类型的列表。

    .. py:method:: project(columns)

        在输入数据集上投影某些列。

        从数据集中选择列，并以指定的顺序传输到流水线中。
        其他列将被丢弃。

        **参数：**

        - **columns** (Union[str, list[str]]) - 要投影列的列名列表。

        **返回：**

        ProjectDataset，投影后的数据集对象。

        **样例：**

        >>> # dataset是Dataset对象的实例
        >>> columns_to_project = ["column3", "column1", "column2"]
        >>>
        >>> # 创建一个数据集，无论列的原始顺序如何，依次包含column3, column1, column2。
        >>> dataset = dataset.project(columns=columns_to_project)

    .. py:method:: rename(input_columns, output_columns)

        重命名输入数据集中的列。

        **参数：**

        - **input_columns** (Union[str, list[str]]) - 输入列的列名列表。
        - **output_columns** (Union[str, list[str]]) - 输出列的列名列表。

        **返回：**

        RenameDataset，重命名后数据集对象。

        **样例：**

        >>> # dataset是Dataset对象的实例
        >>> input_columns = ["input_col1", "input_col2", "input_col3"]
        >>> output_columns = ["output_col1", "output_col2", "output_col3"]
        >>>
        >>> # 创建一个数据集，其中input_col1重命名为output_col1，
        >>> # input_col2重命名为output_col2，input_col3重命名
        >>> # 为output_col3。
        >>> dataset = dataset.rename(input_columns=input_columns, output_columns=output_columns)

    .. py:method:: repeat(count=None)

        重复此数据集 `count` 次。如果count为None或-1，则无限重复。

        .. note::
            repeat和batch的顺序反映了batch的数量。建议：repeat操作在batch操作之后使用。

        **参数：**

        - **count** (int) - 数据集重复的次数（默认为None）。

        **返回：**

        RepeatDataset，重复操作后的数据集对象。

        **样例：**

        >>>  # dataset是Dataset对象的实例
        >>>
        >>> # 创建一个数据集，数据集重复50个epoch。
        >>> dataset = dataset.repeat(50)
        >>>
        >>> # 创建一个数据集，其中每个epoch都是单独打乱的。
        >>> dataset = dataset.shuffle(10)
        >>> dataset = dataset.repeat(50)
        >>>
        >>> # 创建一个数据集，打乱前先将数据集重复
        >>> # 50个epoch。shuffle算子将
        >>> # 整个50个epoch视作一个大数据集。
        >>> dataset = dataset.repeat(50)
        >>> dataset = dataset.shuffle(10)

    .. py:method:: reset()

        重置下一个epoch的数据集。

    .. py:method:: save(file_name, num_files=1, file_type='mindrecord')

        将流水线正在处理的数据保存为通用的数据集格式。支持的数据集格式：'mindrecord'。

        将数据保存为'mindrecord'格式时存在隐式类型转换。转换表展示如何执行类型转换。

        .. list-table:: 保存为'mindrecord'格式时的隐式类型转换
           :widths: 25 25 50
           :header-rows: 1

           * - 'dataset'类型
             - 'mindrecord'类型
             - 详细
           * - bool
             - None
             - 不支持
           * - int8
             - int32
             -
           * - uint8
             - bytes(1D uint8)
             - Drop dimension
           * - int16
             - int32
             -
           * - uint16
             - int32
             -
           * - int32
             - int32
             -
           * - uint32
             - int64
             -
           * - int64
             - int64
             -
           * - uint64
             - None
             - 不支持
           * - float16
             - float32
             -
           * - float32
             - float32
             -
           * - float64
             - float64
             -
           * - string
             - string
             - 不支持多维字符串

        .. note::
            1. 如需按顺序保存示例，请将数据集的shuffle设置为False，将 `num_files` 设置为1。
            2. 在调用函数之前，不要使用batch算子、repeat算子或具有随机属性的数据增强的map算子。
            3. 当数据的维度可变时，只支持1维数组或者在0维变化的多维数组。
            4. 不支持DE_UINT64类型、多维的DE_UINT8类型、多维DE_STRING类型。

        **参数：**

        - **file_name** (str) - 数据集文件的路径。
        - **num_files** (int, optional) - 数据集文件的数量（默认为1）。
        - **file_type** (str, optional) - 数据集格式（默认为'mindrecord'）。

    .. py:method:: set_dynamic_columns(columns=None)

        设置源数据的动态shape信息，需要在定义数据处理流水线后设置。

        **参数：**

        - **columns** (dict) - 包含数据集中每列shape信息的字典。shape[i]为 `None` 表示shape[i]的数据长度是动态的。

    .. py:method:: shuffle(buffer_size)

        使用以下策略随机打乱此数据集的行：

        1. 生成一个shuffle缓冲区包含buffer_size条数据行。

        2. 从shuffle缓冲区中随机选择一个元素，作为下一行传播到子节点。

        3. 从父节点获取下一行（如果有的话），并将其放入shuffle缓冲区中。

        4. 重复步骤2和3，直到打乱缓冲区中没有数据行为止。

        可以提供随机种子，在第一个epoch中使用。在随后的每个epoch，种子都会被设置成一个新产生的随机值。

        **参数：**

        - **buffer_size** (int) - 用于shuffle的缓冲区大小（必须大于1）。将buffer_size设置为等于数据集大小将导致在全局shuffle。

        **返回：**

        ShuffleDataset，打乱后的数据集对象。

        **异常：**

        - **RuntimeError** - 打乱前存在同步操作。

        **样例：**

        >>>  # dataset是Dataset对象的实例
        >>> # 可以选择设置第一个epoch的种子
        >>> ds.config.set_seed(58)
        >>> # 使用大小为4的shuffle缓冲区创建打乱后的数据集。
        >>> dataset = dataset.shuffle(4)

    .. py:method:: skip(count)

        跳过此数据集的前N个元素。

        **参数：**

        - **count** (int) - 要跳过的数据集中的元素个数。

        **返回：**

        SkipDataset，减去跳过的行的数据集对象。

        **样例：**

        >>> # dataset是Dataset对象的实例
        >>> # 创建一个数据集，跳过前3个元素
        >>> dataset = dataset.skip(3)

    .. py:method:: split(sizes, randomize=True)

        将数据集拆分为多个不重叠的数据集。

        这是一个通用拆分函数，可以被数据处理流水线中的任何算子调用。
        还有如果直接调用ds.split，其中 ds 是一个 MappableDataset，它将被自动调用。

        **参数：**

        - **sizes** (Union[list[int], list[float]]) - 如果指定了一列整数[s1, s2, …, sn]，数据集将被拆分为n个大小为s1、s2、...、sn的数据集。如果所有输入大小的总和不等于原始数据集大小，则报错。如果指定了一列浮点数[f1, f2, …, fn]，则所有浮点数必须介于0和1之间，并且总和必须为1，否则报错。数据集将被拆分为n个大小为round(f1*K)、round(f2*K)、...、round(fn*K)的数据集，其中K是原始数据集的大小。

            如果舍入后：

                - 任何大小等于0，都将发生错误。
                - 如果拆分大小的总和<K，K - sigma(round(fi * k))的差值将添加到第一个子数据集。
                - 如果拆分大小的总和>K，sigma(round(fi * K)) - K的差值将从第一个足够大的拆分子集中删除，删除差值后至少有1行。

        - **randomize** (bool, optional) - 确定是否随机拆分数据（默认为True）。如果为True，则数据集将被随机拆分。否则，将使用数据集中的连续行创建每个拆分子集。

        .. note::
            1. 如果要调用 split，则无法对数据集进行分片。
            2. 强烈建议不要对数据集进行打乱，而是使用随机化（randomize=True）。对数据集进行打乱的结果具有不确定性，每个拆分子集中的数据在每个epoch可能都不同。

        **异常：**

        - **RuntimeError** - get_dataset_size返回None或此数据集不支持。
        - **RuntimeError** - sizes是整数列表，并且size中所有元素的总和不等于数据集大小。
        - **RuntimeError** - sizes是float列表，并且计算后存在大小为0的拆分子数据集。
        - **RuntimeError** - 数据集在调用拆分之前已进行分片。
        - **ValueError** - sizes是float列表，且并非所有float数都在0和1之间，或者float数的总和不等于1。

        **返回：**

        tuple(Dataset)，拆分后子数据集对象的元组。

        **样例：**

        >>> # TextFileDataset不是可映射dataset，因此将调用通用拆分函数。
        >>> # 由于许多数据集默认都打开了shuffle，如需调用拆分函数，请将shuffle设置为False。
        >>> dataset = ds.TextFileDataset(text_file_dataset_dir, shuffle=False)
        >>> train_dataset, test_dataset = dataset.split([0.9, 0.1])

    .. py:method:: sync_update(condition_name, num_batch=None, data=None)

        释放阻塞条件并使用给定数据触发回调函数。

        **参数：**

        - **condition_name** (str) - 用于切换发送下一行数据的条件名称。
        - **num_batch** (Union[int, None]) - 释放的batch（row）数。当 `num_batch` 为None时，将默认为 `sync_wait` 算子指定的值（默认为None）。
        - **data** (Any) - 用户自定义传递给回调函数的数据（默认为None）。

    .. py:method:: sync_wait(condition_name, num_batch=1, callback=None)

        向输入数据集添加阻塞条件。 将应用同步操作。

        **参数：**

        - **condition_name** (str) - 用于切换发送下一行的条件名称。
        - **num_batch** (int) - 每个epoch开始时无阻塞的batch数。
        - **callback** (function) -  `sync_update` 中将调用的回调函数。

        **返回：**

        SyncWaitDataset，添加了阻塞条件的数据集对象。

        **异常：**

        - **RuntimeError** - 条件名称已存在。

        **样例：**

        >>> import numpy as np
        >>> def gen():
        ...     for i in range(100)：
        ...         yield (np.array(i),)
        >>>
        >>> class Augment:
        ...     def __init__(self, loss)：
        ...         self.loss = loss
        ...
        ...     def preprocess(self, input_)：
        ...         return input_
        ...
        ...     def update(self, data)：
        ...         self.loss = data["loss"]
        >>>
        >>> batch_size = 4
        >>> dataset = ds.GeneratorDataset(gen, column_names=["input"])
        >>>
        >>> aug = Augment(0)
        >>> dataset = dataset.sync_wait(condition_name="policy", callback=aug.update)
        >>> dataset = dataset.map(operations=[aug.preprocess], input_columns=["input"])
        >>> dataset = dataset.batch(batch_size)
        >>> count = 0
        >>> for data in dataset.create_dict_iterator(num_epochs=1, output_numpy=True)：
        ...     assert data["input"][0] == count
        ...     count += batch_size
        ...     data = {"loss": count}
        ...     dataset.sync_update(condition_name="policy", data=data)

    .. py:method:: take(count=-1)

        从数据集中获取最多给定数量的元素。

        .. note::
            1. 如果count大于数据集中的元素数或等于-1，则取数据集中的所有元素。
            2. take和batch操作顺序很重要，如果take在batch操作之前，则取给定行数；否则取给定batch数。

        **参数：**

        - **count** (int, optional) - 要从数据集中获取的元素数（默认为-1）。

        **返回：**

        TakeDataset，取出指定数目的数据集对象。

        **样例：**

        >>> # dataset是Dataset对象的实例。
        >>> # 创建一个数据集，包含50个元素。
        >>> dataset = dataset.take(50)

    .. py:method:: to_device(send_epoch_end=True, create_data_info_queue=False)

        将数据从CPU传输到GPU、Ascend或其他设备。

        **参数：**

        - **send_epoch_end** (bool, optional) - 是否将end of sequence发送到设备（默认为True）。
        - **create_data_info_queue** (bool, optional) - 是否创建存储数据类型和shape的队列（默认值为False）。

        .. note::
            如果设备为Ascend，则逐个传输数据。每次传输的数据最大限制为256M。

        **返回：**

        TransferDataset，用于传输的数据集对象。

        **异常：**

        - **RuntimeError** - 如果提供了分布式训练的文件路径但读取失败。

    .. py:method:: to_json(filename='')

        将数据处理流水线序列化为JSON字符串，如果提供了文件名，则转储到文件中。

        **参数：**

        - **filename** (str) - 另存为JSON格式的文件名。

        **返回：**

        str，流水线的JSON字符串。
