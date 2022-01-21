    .. py:method:: close_pool()

        关闭数据集对象中的多进程池。如果您熟悉多进程库，可以将此视为进程池对象的析构函数。

    .. py:method:: concat(datasets)

        对传入的多个数据集对象进行拼接操作，也可以使用"+"运算符来进行数据集进行拼接。

        .. note::
            用于拼接的多个数据集对象，每个数据集对象的列名、每列数据的维度（rank）和数据类型必须相同。

        **参数：**

        - **datasets** (Union[list, Dataset]) - 与当前数据集对象拼接的数据集对象列表或单个数据集对象。

        **返回：**

        Dataset，拼接后的数据集对象。

    .. py:method:: create_dict_iterator(num_epochs=-1, output_numpy=False)

        基于数据集对象创建迭代器，输出的数据为字典类型。

        **参数：**

        - **num_epochs** (int, 可选) - 迭代器可以迭代的最大次数。默认值：-1，迭代器可以迭代无限次。
        - **output_numpy** (bool, 可选) - 输出的数据是否转为NumPy类型。如果为False，迭代器输出的每列数据类型为MindSpore.Tensor，否则为NumPy。默认值：False。

        **返回：**

        DictIterator，基于数据集对象创建的字典迭代器。

    .. py:method:: create_tuple_iterator(columns=None, num_epochs=-1, output_numpy=False, do_copy=True)

        基于数据集对象创建迭代器，输出数据为ndarray组成的列表。

        可以通过参数 `columns` 指定输出的所有列名及列的顺序。如果columns未指定，列的顺序将保持不变。

        **参数：**

        - **columns** (list[str], 可选) - 用于指定输出的数据列和列的顺序。默认值：None，输出所有数据列。
        - **num_epochs** (int, 可选) - 迭代器可以迭代的最大次数。默认值：-1，迭代器可以迭代无限次。
        - **output_numpy** (bool, 可选) - 输出的数据是否转为NumPy类型。如果为False，迭代器输出的每列数据类型为MindSpore.Tensor，否则为NumPy。默认值：False。
        - **do_copy** (bool, 可选) - 当参数 `output_numpy` 为False，即输出数据类型为mindspore.Tensor时，可以将此参数指定为False以减少拷贝，获得更好的性能。默认值：True。

        **返回：**

        TupleIterator，基于数据集对象创建的元组迭代器。

    .. py:method:: device_que(send_epoch_end=True, create_data_info_queue=False)

        将数据异步传输到Ascend/GPU设备上。

        **参数：**

        - **send_epoch_end** (bool, 可选) - 数据发送完成后是否发送结束标识到设备上，默认值：True。
        - **create_data_info_queue** (bool, 可选) - 是否创建一个队列，用于存储每条数据的数据类型和shape。默认值：False，不创建。

        .. note::
            如果设备类型为Ascend，每次传输的数据大小限制为256MB。

        **返回：**

        Dataset，用于帮助发送数据到设备上的数据集对象。


    .. py:method:: dynamic_min_max_shapes()

        当数据集对象中的数据shape不唯一（动态shape）时，获取数据的最小shape和最大shape。

        **返回：**

        两个列表代表最小shape和最大shape，每个列表中的shape按照数据列的顺序排列。


    .. py:method:: filter(predicate, input_columns=None, num_parallel_workers=None)

        通过自定义判断条件对数据集对象中的数据进行过滤。

        **参数：**

        - **predicate** (callable) - Python可调用对象。要求该对象接收n个入参，用于指代每个数据列的数据，最后返回值一个bool值。
          如果返回值为False，则表示过滤掉该条数据。注意n的值与参数 `input_columns` 表示的输入列数量一致。
        - **input_columns** (Union[str, list[str]], 可选) - `filter` 操作的输入数据列。默认值：None，`predicate` 将应用于数据集中的所有列。
        - **num_parallel_workers** (int, 可选) - 指定 `filter` 操作的并发线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。

        **返回：**

        Dataset，执行给定筛选过滤操作的数据集对象。


    .. py:method:: flat_map(func)

        对数据集对象中每一条数据执行给定的数据处理，并将结果展平。

        **参数：**

        - **func** (function) - 数据处理函数，要求输入必须为一个'ndarray'，返回值是一个'Dataset'对象。

        **返回：**

        执行给定操作后的数据集对象。

        **异常：**

        - **TypeError** - `func` 不是函数。
        - **TypeError** - `func` 的返回值不是数据集对象。

    .. py:method:: get_batch_size()

        获得数据集对象定义的批处理大小，即一个批处理数据中包含的数据条数。

        **返回：**

        int，一个批处理数据中包含的数据条数。

    .. py:method:: get_class_indexing()

        返回类别索引。

        **返回：**

        dict，描述类别名称到索引的键值对映射关系，通常为str-to-int格式。针对COCO数据集，类别名称到索引映射关系描述形式为str-to-list<int>格式，列表中的第二个数字表示超级类别。


    .. py:method:: get_col_names()

        返回数据集对象中包含的列名。

        **返回：**

        list，数据集中所有列名组成列表。

    .. py:method:: get_dataset_size()

        返回一个epoch中的batch数。

        **返回：**

        int，batch的数目。

    .. py:method:: get_repeat_count()

        获取 `RepeatDataset` 中定义的repeat操作的次数。默认值：1。

        **返回：**

        int，repeat操作的次数。

    .. py:method:: input_indexs
        :property:

        获取input index信息。

        **返回：**

        input index信息的元组。

    .. py:method:: map(operations, input_columns=None, output_columns=None, column_order=None, num_parallel_workers=None, python_multiprocessing=False, cache=None, callbacks=None, max_rowsize=16, offload=None)

        给定一组数据增强列表，按顺序将数据增强作用在数据集对象上。

        每个数据增强操作将数据集对象中的一个或多个数据列作为输入，将数据增强的结果输出为一个或多个数据列。
        第一个数据增强操作将 `input_columns` 中指定的列作为输入。
        如果数据增强列表中存在多个数据增强操作，则上一个数据增强的输出列将作为下一个数据增强的输入列。

        最后一个数据增强的输出列的列名由 `output_columns` 指定，如果没有指定 `output_columns` ，输出列名与 `input_columns` 一致。

        **参数：**

        - **operations** (Union[list[TensorOp], list[functions]]) - 一组数据增强操作，支持数据集增强算子或者用户自定义的Python Callable对象。map操作将按顺序将一组数据增强作用在数据集对象上。
        - **input_columns** (Union[str, list[str]], 可选) - 第一个数据增强操作的输入数据列。此列表的长度必须与 `operations` 列表中第一个数据增强的预期输入列数相匹配。默认值：None。表示所有数据列都将传递给第一个数据增强操作。
        - **output_columns** (Union[str, list[str]], 可选) - 最后一个数据增强操作的输出数据列。如果 `input_columns` 长度不等于 `output_columns` 长度，则必须指定此参数。列表的长度必须必须与最后一个数据增强的输出列数相匹配。默认值：None，输出列将与输入列具有相同的名称。
        - **column_order** (Union[str, list[str]], 可选) - 指定传递到下一个数据集操作的数据列的顺序。如果 `input_columns` 长度不等于 `output_columns` 长度，则必须指定此参数。 注意：参数的列名不限定在 `input_columns` 和 `output_columns` 中指定的列，也可以是上一个操作输出的未被处理的数据列。默认值：None，按照原输入顺序排列。
        - **num_parallel_workers** (int, 可选) - 指定map操作的多进程/多线程并发数，加快处理速度。默认值：None，将使用 `set_num_parallel_workers` 设置的并发数。
        - **python_multiprocessing** (bool, 可选) - 启用Python多进程模式加速map操作。当传入的 `operations` 计算量很大时，开启此选项可能会有较好效果。默认值：False。
        - **cache** (DatasetCache, 可选) - 单节点数据缓存服务，用于加快数据集处理，详情请阅读 `单节点数据缓存 <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/cache.html>`_ 。默认值：None，不使用缓存。
        - **callbacks** (DSCallback, list[DSCallback], 可选) - 要调用的Dataset回调函数列表。默认值：None。
        - **max_rowsize** (int, 可选) - 指定在多进程之间复制数据时，共享内存分配的最大空间，仅当 `python_multiprocessing` 为True时，该选项有效。默认值：16，数量级为MB。
        - **offload** (bool, 可选) - 是否进行异构硬件加速，详情请阅读 `数据准备异构加速 <https://www.mindspore.cn/docs/programming_guide/zh-CN/master/enable_dataset_offload.html>`_ 。默认值：None。

        .. note::
            - `operations` 参数主要接收 `mindspore.dataset` 模块中c_transforms、py_transforms算子，以及用户定义的Python函数(PyFuncs)。
            - 不要将 `mindspore.nn` 和 `mindspore.ops` 或其他的网络计算算子添加到 `operations` 中。

        **返回：**

        MapDataset，map操作后的数据集。

    .. py:method:: num_classes()

        获取数据集对象中所有样本的类别数目。

        **返回：**

        int，类别的数目。

    .. py:method:: output_shapes()

        获取数据集对象中每列数据的shape。

        **返回：**

        list，每列数据的shape列表。

    .. py:method:: output_types()

        获取数据集对象中每列数据的数据类型。

        **返回：**

        list，每列数据的数据类型列表。

    .. py:method:: project(columns)

        从数据集对象中选择需要的列，并按给定的列名的顺序进行排序，
        未指定的数据列将被丢弃。

        **参数：**

        - **columns** (Union[str, list[str]]) - 要选择的数据列的列名列表。

        **返回：**

        ProjectDataset，project操作后的数据集对象。

    .. py:method:: rename(input_columns, output_columns)

        对数据集对象按指定的列名进行重命名。

        **参数：**

        - **input_columns** (Union[str, list[str]]) - 待重命名的列名列表。
        - **output_columns** (Union[str, list[str]]) - 重命名后的列名列表。

        **返回：**

        RenameDataset，rename操作后的数据集对象。

    .. py:method:: repeat(count=None)

        重复此数据集 `count` 次。如果 `count` 为None或-1，则无限重复。

        .. note::
            repeat和batch的顺序反映了batch的数量。建议：repeat操作在batch操作之后使用。

        **参数：**

        - **count** (int) - 数据集重复的次数。默认值：None。

        **返回：**

        RepeatDataset，repeat操作后的数据集对象。

    .. py:method:: reset()

        重置下一个epoch的数据集对象。

    .. py:method:: save(file_name, num_files=1, file_type='mindrecord')

        将数据处理管道中正处理的数据保存为通用的数据集格式。支持的数据集格式：'mindrecord'。

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
             - bytes
             - 丢失维度信息
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
            1. 如需按顺序保存数据，将数据集的 `shuffle` 设置为False，将 `num_files` 设置为1。
            2. 在执行保存操作之前，不要使用batch操作、repeat操作或具有随机属性的数据增强的map操作。
            3. 当数据的维度可变时，只支持1维数组或者在第0维变化的多维数组。
            4. 不支持UINT64类型、多维的UINT8类型、多维STRING类型。

        **参数：**

        - **file_name** (str) - 数据集文件的路径。
        - **num_files** (int, 可选) - 数据集文件的数量，默认值：1。
        - **file_type** (str, 可选) - 数据集格式，默认值：'mindrecord'。

    .. py:method:: set_dynamic_columns(columns=None)

        设置数据集的动态shape信息，需要在定义好完整的数据处理管道后进行设置。

        **参数：**

        - **columns** (dict) - 包含数据集中每列shape信息的字典。shape[i]为 `None` 表示shape[i]的数据长度是动态的。

    .. py:method:: shuffle(buffer_size)

        使用以下策略混洗此数据集的行：

        1. 生成一个混洗缓冲区包含 `buffer_size` 条数据行。

        2. 从混洗缓冲区中随机选择一个数据行，传递给下一个操作。

        3. 从上一个操作获取下一个数据行（如果有的话），并将其放入混洗缓冲区中。

        4. 重复步骤2和3，直到混洗缓冲区中没有数据行为止。

        在第一个epoch中可以通过 `dataset.config.set_seed` 来设置随机种子，在随后的每个epoch，种子都会被设置成一个新产生的随机值。

        **参数：**

        - **buffer_size** (int) - 用于混洗的缓冲区大小（必须大于1）。将 `buffer_size` 设置为数据集大小将进行全局混洗。

        **返回：**

        ShuffleDataset，混洗后的数据集对象。

        **异常：**

        - **RuntimeError** - 混洗前存在通过 `dataset.sync_wait` 进行同步操作。

    .. py:method:: skip(count)

        跳过此数据集对象的前 `count` 条数据。

        **参数：**

        - **count** (int) - 要跳过数据的条数。

        **返回：**

        SkipDataset，跳过指定条数据后的数据集对象。

    .. py:method:: split(sizes, randomize=True)

        将数据集拆分为多个不重叠的子数据集。

        **参数：**

        - **sizes** (Union[list[int], list[float]]) - 如果指定了一列整数[s1, s2, …, sn]，数据集将被拆分为n个大小为s1、s2、...、sn的数据集。如果所有输入大小的总和不等于原始数据集大小，则报错。如果指定了一列浮点数[f1, f2, …, fn]，则所有浮点数必须介于0和1之间，并且总和必须为1，否则报错。数据集将被拆分为n个大小为round(f1*K)、round(f2*K)、...、round(fn*K)的数据集，其中K是原始数据集的大小。

            如果round四舍五入计算后：

                - 任何子数据集的的大小等于0，都将发生错误。
                - 如果子数据集大小的总和小于K，K - sigma(round(fi * k))的值将添加到第一个子数据集，sigma为求和操作。
                - 如果子数据集大小的总和大于K，sigma(round(fi * K)) - K的值将从第一个足够大的子数据集中删除，且删除后的子数据集大小至少大于1。

        - **randomize** (bool, 可选) - 确定是否随机拆分数据，默认值：True，数据集将被随机拆分。否则将按顺序拆分为多个不重叠的子数据集。

        .. note::
            1. 如果进行拆分操作的数据集对象为MappableDataset类型，则将自动调用一个优化后的split操作。
            2. 如果进行split操作，则不应对数据集对象进行分片操作（如指定num_shards或使用DistributerSampler）。相反，如果创建一个DistributerSampler，并在split操作拆分后的子数据集对象上进行分片操作，强烈建议在每个子数据集上设置相同的种子，否则每个分片可能不是同一个子数据集的一部分（请参见示例）。
            3. 强烈建议不要对数据集进行混洗，而是使用随机化（randomize=True）。对数据集进行混洗的结果具有不确定性，每个拆分后的子数据集中的数据在每个epoch可能都不同。

        **异常：**

        - **RuntimeError** - 数据集对象不支持 `get_dataset_size` 或者 `get_dataset_size` 返回None。
        - **RuntimeError** - sizes是整数列表，并且size中所有元素的总和不等于数据集大小。
        - **RuntimeError** - sizes是float列表，并且计算后存在大小为0的拆分子数据集。
        - **RuntimeError** - 数据集对象在调用拆分之前已进行分片。
        - **ValueError** - sizes是float列表，且并非所有float数都在0和1之间，或者float数的总和不等于1。

        **返回：**

        tuple(Dataset)，split操作后子数据集对象的元组。

    .. py:method:: sync_update(condition_name, num_batch=None, data=None)

        释放阻塞条件并使用给定数据触发回调函数。

        **参数：**

        - **condition_name** (str) - 用于触发发送下一个数据行的条件名称。
        - **num_batch** (Union[int, None]) - 释放的batch（row）数。当 `num_batch` 为None时，将默认为 `sync_wait`  操作指定的值，默认值：None。
        - **data** (Any) - 用户自定义传递给回调函数的数据，默认值：None。

    .. py:method:: sync_wait(condition_name, num_batch=1, callback=None)

        为同步操作在数据集对象上添加阻塞条件。

        **参数：**

        - **condition_name** (str) - 用于触发发送下一行数据的条件名称。
        - **num_batch** (int) - 每个epoch开始时无阻塞的batch数。
        - **callback** (function) -  `sync_update` 操作中将调用的回调函数。

        **返回：**

        SyncWaitDataset，添加了阻塞条件的数据集对象。

        **异常：**

        - **RuntimeError** - 条件名称已存在。

    .. py:method:: take(count=-1)

        从数据集中获取最多 `count` 的元素。

        .. note::
            1. 如果 `count` 大于数据集中的数据条数或等于-1，则取数据集中的所有数据。
            2. take和batch操作顺序很重要，如果take在batch操作之前，则取给定条数，否则取给定batch数。

        **参数：**

        - **count** (int, 可选) - 要从数据集对象中获取的数据条数，默认值：-1，获取所有数据。

        **返回：**

        TakeDataset，take操作后的数据集对象。

    .. py:method:: to_device(send_epoch_end=True, create_data_info_queue=False)

        将数据从CPU传输到GPU、Ascend或其他设备。

        **参数：**

        - **send_epoch_end** (bool, 可选) - 是否将epoch结束符 `end_of_sequence` 发送到设备，默认值：True。
        - **create_data_info_queue** (bool, 可选) - 是否创建存储数据类型和shape的队列，默认值：False。

        .. note::
            如果设备为Ascend，则逐个传输数据。每次数据传输的限制为256M。

        **返回：**

        TransferDataset，用于传输的数据集对象。

        **异常：**

        - **RuntimeError** - 如果提供了分布式训练的文件路径但读取失败。

    .. py:method:: to_json(filename='')

        将数据处理管道序列化为JSON字符串，如果提供了文件名，则转储到文件中。

        **参数：**

        - **filename** (str) - 保存JSON文件的路径（包含文件名）。

        **返回：**

        str，数据处理管道序列化后的JSON字符串。
