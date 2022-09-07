.. py:method:: close_pool()

    关闭数据集对象中的多进程池。如果您熟悉多进程库，可以将此视为进程池对象的析构函数。

    .. note::
        该接口在将来会被删除或不可见，不建议用户调用该接口。

.. py:method:: concat(datasets)

    对传入的多个数据集对象进行拼接操作。可以使用"+"运算符来进行数据集进行拼接。

    .. note::
        用于拼接的多个数据集对象，每个数据集对象的列名、每列数据的维度（rank）和数据类型必须相同。

    参数：
        - **datasets** (Union[list, Dataset]) - 与当前数据集对象拼接的数据集对象列表或单个数据集对象。

    返回：
        Dataset，拼接后的数据集对象。

.. py:method:: create_dict_iterator(num_epochs=-1, output_numpy=False)

    基于数据集对象创建迭代器。输出的数据为字典类型。

    参数：
        - **num_epochs** (int, 可选) - 迭代器可以迭代的最大次数。默认值：-1，迭代器可以迭代无限次。
        - **output_numpy** (bool, 可选) - 输出的数据是否转为NumPy类型。如果为False，迭代器输出的每列数据类型为MindSpore.Tensor，否则为NumPy。默认值：False。

    返回：
        DictIterator，基于数据集对象创建的字典迭代器。

.. py:method:: create_tuple_iterator(columns=None, num_epochs=-1, output_numpy=False, do_copy=True)

    基于数据集对象创建迭代器。输出数据为 `numpy.ndarray` 组成的列表。

    可以通过参数 `columns` 指定输出的所有列名及列的顺序。如果columns未指定，列的顺序将保持不变。

    参数：
        - **columns** (list[str], 可选) - 用于指定输出的数据列和列的顺序。默认值：None，输出所有数据列。
        - **num_epochs** (int, 可选) - 迭代器可以迭代的最大次数。默认值：-1，迭代器可以迭代无限次。
        - **output_numpy** (bool, 可选) - 输出的数据是否转为NumPy类型。如果为False，迭代器输出的每列数据类型为MindSpore.Tensor，否则为NumPy。默认值：False。
        - **do_copy** (bool, 可选) - 当参数 `output_numpy` 为False，即输出数据类型为mindspore.Tensor时，可以将此参数指定为False以减少拷贝，获得更好的性能。默认值：True。

    返回：
        TupleIterator，基于数据集对象创建的元组迭代器。

.. py:method:: device_que(send_epoch_end=True, create_data_info_queue=False)

    将数据异步传输到Ascend/GPU设备上。

    参数：
        - **send_epoch_end** (bool, 可选) - 数据发送完成后是否发送结束标识到设备上，默认值：True。
        - **create_data_info_queue** (bool, 可选) - 是否创建一个队列，用于存储每条数据的数据类型和shape。默认值：False，不创建。

    .. note::
        如果设备类型为Ascend，数据的特征将被逐一传输。每次传输的数据大小限制为256MB。

    返回：
        Dataset，用于帮助发送数据到设备上的数据集对象。

.. py:method:: dynamic_min_max_shapes()

    当数据集对象中的数据shape不唯一（动态shape）时，获取数据的最小shape和最大shape。

    返回：
        两个列表代表最小shape和最大shape，每个列表中的shape按照数据列的顺序排列。

.. py:method:: filter(predicate, input_columns=None, num_parallel_workers=None)

    通过自定义判断条件对数据集对象中的数据进行过滤。

    参数：
        - **predicate** (callable) - Python可调用对象。要求该对象接收n个入参，用于指代每个数据列的数据，最后返回值一个bool值。
          如果返回值为False，则表示过滤掉该条数据。注意n的值与参数 `input_columns` 表示的输入列数量一致。
        - **input_columns** (Union[str, list[str]], 可选) - `filter` 操作的输入数据列。默认值：None，`predicate` 将应用于数据集中的所有列。
        - **num_parallel_workers** (int, 可选) - 指定 `filter` 操作的并发线程数。默认值：None，使用mindspore.dataset.config中配置的线程数。

    返回：
        Dataset，执行给定筛选过滤操作的数据集对象。

.. py:method:: flat_map(func)

    对数据集对象中每一条数据执行给定的数据处理，并将结果展平。

    参数：
        - **func** (function) - 数据处理函数，要求输入必须为一个 `numpy.ndarray` ，返回值是一个 `Dataset` 对象。

    返回：
        执行给定操作后的数据集对象。

    异常：
        - **TypeError** - `func` 不是函数。
        - **TypeError** - `func` 的返回值不是 `Dataset` 对象。

.. py:method:: get_batch_size()

    获得数据集对象定义的批处理大小，即一个批处理数据中包含的数据条数。

    返回：
        int，一个批处理数据中包含的数据条数。

.. py:method:: get_class_indexing()

    返回类别索引。

    返回：
        dict，描述类别名称到索引的键值对映射关系，通常为str-to-int格式。针对COCO数据集，类别名称到索引映射关系描述形式为str-to-list<int>格式，列表中的第二个数字表示超类别。

.. py:method:: get_col_names()

    返回数据集对象中包含的列名。

    返回：
        list，数据集中所有列名组成列表。

.. py:method:: get_dataset_size()

    返回一个epoch中的batch数。

    返回：
        int，batch的数目。

.. py:method:: get_repeat_count()

    获取 `RepeatDataset` 中定义的repeat操作的次数，默认值：1。

    返回：
        int，repeat操作的次数。

.. py:method:: input_indexs
    :property:

    获取/设置数据列索引，它表示使用下沉模式时数据列映射至网络中的对应关系。

    返回：
        int，数据集的input index信息。
