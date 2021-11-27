.. py:method:: batch(batch_size, drop_remainder=False, num_parallel_workers=None, per_batch_map=None, input_columns=None, output_columns=None, column_order=None, pad_info=None, python_multiprocessing=False)

    将dataset中连续 `batch_size` 行数据合并为一个批处理数据。

    对一个批处理数据执行给定操作与对条数据进行给定操作用法一致。对于任意列，batch操作要求该列中的各条数据shape必须相同。如果给定可执行函数 `per_batch_map` ，它将作用于批处理后的数据。

    .. note::
        执行 `repeat` 和 `batch` 操作的顺序，会影响数据批次的数量及 `per_batch_map` 操作。建议在batch操作完成后执行repeat操作。

   ** 参数：**

    - **batch_size** (int or function)：每个批处理数据包含的条数。参数需要是int或可调用对象，该对象接收1个参数，即BatchInfo。
    - **drop_remainder** (bool, optional)：是否删除最后一个数据条数小于批处理大小的batch（默认值为False）。如果为True，并且最后一个批次中数据行数少于 `batch_size`，则这些数据将被丢弃，不会传递给后续的操作。
    - **num_parallel_workers** (int, optional)：用于进行batch操作的的线程数（threads），默认值为None。
    - **per_batch_map** (callable, optional)：是一个以(list[Tensor], list[Tensor], ..., BatchInfo)作为输入参数的可调用对象。每个list[Tensor]代表给定列上的一批Tensor。入参中list[Tensor]的个数应与 `input_columns` 中传入列名的数量相匹配。该可调用对象的最后一个参数始终是BatchInfo对象。`per_batch_map`应返回(list[Tensor], list[Tensor], ...)。其出中list[Tensor]的个数应与输入相同。如果输出列数与输入列数不一致，则需要指定 `output_columns`。        - **input_columns** (Union[str, list[str]], optional)：由输入列名组成的列表。如果 `per_batch_map` 不为None，列表中列名的个数应与 `per_batch_map` 中包含的列数匹配（默认为None）。
    - **output_columns** (Union[str, list[str]], optional)：当前操作所有输出列的列名列表。如果len(input_columns) != len(output_columns)，则此参数必须指定。此列表中列名的数量必须与给定操作的输出列数相匹配（默认为None，输出列将与输入列具有相同的名称）。
    - **column_order** (Union[str, list[str]], optional)：指定整个数据集对象中包含的所有列名的顺序。如果len(input_column) != len(output_column)，则此参数必须指定。 注意：这里的列名不仅仅是在 `input_columns` 和 `output_columns` 中指定的列。
    - **pad_info** (dict, optional)：用于对给定列进行填充。例如 `pad_info={"col1":([224,224],0)}` ，则将列名为"col1"的列填充到大小为[224,224]的张量，并用0填充缺失的值（默认为None)。
    - **python_multiprocessing** (bool, optional)：针对 `per_batch_map` 函数，使用Python多进执行的方式进行调用。如果函数计算量大，开启这个选项可能会很有帮助（默认值为False）。

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