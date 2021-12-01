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
        - **python_multiprocessing** (bool, optional) - 将Python operations委托给多个工作进程进行并行处理。如果Python operations计算量很大，此选项可能会很有用（默认为False）。
        - **cache** (DatasetCache, optional) - 使用Tensor缓存服务加快数据集处理速度（默认为None，即不使用缓存）。
        - **callbacks** (DSCallback, list[DSCallback], optional) - 要调用的Dataset回调函数列表（默认为None）。

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
        
        **columns** (Union[str, list[str]]) - 要投影列的列名列表。

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

        **count** (int) - 数据集重复的次数（默认为None）。

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
        
    ..py:method:: reset()
        
        重置下一个epoch的数据集。

    ..py:method:: save(file_name, num_files=1, file_type='mindrecord')

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

    ..py:method:: set_dynamic_columns(columns=None)

        设置源数据的动态shape信息，需要在定义数据处理流水线后设置。

        **参数：**
        
        **columns** (dict) - 包含数据集中每列shape信息的字典。shape[i]为 `None` 表示shape[i]的数据长度是动态的。
        
    ..py:method:: shuffle(buffer_size)

        使用以下策略随机打乱此数据集的行：

        1. 生成一个shuffle缓冲区包含buffer_size条数据行。

        2. 从shuffle缓冲区中随机选择一个元素，作为下一行传播到子节点。 

        3. 从父节点获取下一行（如果有的话），并将其放入shuffle缓冲区中。

        4. 重复步骤2和3，直到打乱缓冲区中没有数据行为止。

        可以提供随机种子，在第一个epoch中使用。在随后的每个epoch，种子都会被设置成一个新产生的随机值。
        
        **参数：**
        
        **buffer_size** (int) - 用于shuffle的缓冲区大小（必须大于1）。将buffer_size设置为等于数据集大小将导致在全局shuffle。
                
        **返回：**
        
        ShuffleDataset，打乱后的数据集对象。

        **异常：**
        
        **RuntimeError** - 打乱前存在同步操作。

        **样例：**

        >>>  # dataset是Dataset对象的实例
        >>> # 可以选择设置第一个epoch的种子
        >>> ds.config.set_seed(58)
        >>> # 使用大小为4的shuffle缓冲区创建打乱后的数据集。
        >>> dataset = dataset.shuffle(4)
        
    ..py:method:: skip(count)

        跳过此数据集的前N个元素。

        **参数：**

        **count** (int) - 要跳过的数据集中的元素个数。

        **返回：**

        SkipDataset，减去跳过的行的数据集对象。

        **样例：**

        >>> # dataset是Dataset对象的实例
        >>> # 创建一个数据集，跳过前3个元素
        >>> dataset = dataset.skip(3)
        
    ..py:method:: split(sizes, randomize=True)

        将数据集拆分为多个不重叠的数据集。

        这是一个通用拆分函数，可以被数据处理流水线中的任何算子调用。
        还有如果直接调用ds.split，其中 ds 是一个 MappableDataset，它将被自动调用。  

        **参数：**
        
        - **sizes** (Union[list[int], list[float]]) - 如果指定了一列整数[s1, s2, …, sn]，数据集将被拆分为n个大小为s1、s2、...、sn的数据集。如果所有输入大小的总和不等于原始数据集大小，则报错。如果指定了一列浮点数[f1, f2, …, fn]，则所有浮点数必须介于0和1之间，并且总和必须为1，否则报错。数据集将被拆分为n个大小为round(f1*K)、round(f2*K)、...、round(fn*K)的数据集，其中K是原始数据集的大小。
                    
            如果舍入后：

                - 任何大小等于0，都将发生错误。
                - 如果拆分大小的总和<K，K - sigma(round(fi * k))的差值将添加到第一个子数据集。  
                - 如果拆分大小的总和>K，sigma(round(fi * K)) - K的差值将从第一个足够大的拆分子集中删除，删除差值后至少有1行。
                  
        - **randomize** (bool, optional)：确定是否随机拆分数据（默认为True）。如果为True，则数据集将被随机拆分。否则，将使用数据集中的连续行创建每个拆分子集。
                
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
        
    ..py:method:: sync_update(condition_name, num_batch=None, data=None)

        释放阻塞条件并使用给定数据触发回调函数。

        **参数：**

        - **condition_name** (str) - 用于切换发送下一行数据的条件名称。
        - **num_batch** (Union[int, None]) - 释放的batch（row）数。当 `num_batch` 为None时，将默认为 `sync_wait` 算子指定的值（默认为None）。        
        - **data** (Any) - 用户自定义传递给回调函数的数据（默认为None）。
        
    ..py:method:: sync_wait(condition_name, num_batch=1, callback=None)

        向输入数据集添加阻塞条件。 将应用同步操作。

        **参数：**
        
        - **condition_name** (str) - 用于切换发送下一行的条件名称。
        - **num_batch** (int) - 每个epoch开始时无阻塞的batch数。
        - **callback** (function) -  `sync_update` 中将调用的回调函数。

        **返回：**
        
        SyncWaitDataset，添加了阻塞条件的数据集对象。

        **异常：**
        
        **RuntimeError** - 条件名称已存在。

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
        
    ..py:method:: take(count=-1)

        从数据集中获取最多给定数量的元素。

        .. note::
            1. 如果count大于数据集中的元素数或等于-1，则取数据集中的所有元素。     
            2. take和batch操作顺序很重要，如果take在batch操作之前，则取给定行数；否则取给定batch数。
               
        **参数：**
        
        **count** (int, optional) - 要从数据集中获取的元素数（默认为-1）。

        **返回：**
        
        TakeDataset，取出指定数目的数据集对象。

        **样例：**

        >>> # dataset是Dataset对象的实例。
        >>> # 创建一个数据集，包含50个元素。
        >>> dataset = dataset.take(50)
        
    ..py:method:: to_device(send_epoch_end=True, create_data_info_queue=False)

        将数据从CPU传输到GPU、Ascend或其他设备。

        **参数：**
        
        - **send_epoch_end** (bool, optional) - 是否将end of sequence发送到设备（默认为True）。
        - **create_data_info_queue** (bool, optional) - 是否创建存储数据类型和shape的队列（默认为False）。
                
        .. note::
            如果设备为Ascend，则逐个传输数据。每次传输的数据大小限制为256M。
            
        **返回：**
        
        TransferDataset，用于传输的数据集对象。

        **异常：**
        
        **RuntimeError** - 如果提供了分布式训练的文件路径但读取失败。
        
    ..py:method:: to_json(filename='')

        将数据处理流水线序列化为JSON字符串，如果提供了文件名，则转储到文件中。

        **参数：**

        **filename** (str) - 另存为JSON格式的文件名。

        **返回：**
        
        str，流水线的JSON字符串。
        
    ..py:method:: zip(datasets)

        将数据集和输入的数据集或者数据集元组按列进行合并压缩。输入数据集中的列名必须不同。

        **参数：**
        
        **datasets** (Union[tuple, class Dataset]) - 数据集对象的元组或单个数据集对象与当前数据集一起合并压缩。

        **返回：**
            
        ZipDataset，合并压缩后的数据集对象。

        **样例：**
        
        >>> # 创建一个数据集，它将dataset和dataset_1进行合并
        >>> dataset = dataset.zip(dataset_1)
        