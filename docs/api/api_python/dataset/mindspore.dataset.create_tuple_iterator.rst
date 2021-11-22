create_tuple_iterator(columns=None, num_epochs=-1, output_numpy=False, do_copy=True)

    基于数据集对象创建迭代器，输出数据为ndarray组成的列表。

    可以使用columns指定输出的所有列名及列的顺序。如果columns未指定，列的顺序将保持不变。

    参数：
        columns (list[str], optional)：用于指定列顺序的列名列表
            （默认为None，表示所有列）。
        num_epochs (int, optional)：迭代器可以迭代的最多轮次数
            （默认为-1，迭代器可以迭代无限次）。
        output_numpy (bool, optional)：是否输出NumPy数据类型，
            如果output_numpy为False，迭代器输出的每列数据类型为MindSpore.Tensor（默认为False）。
        do_copy (bool, optional)：当输出数据类型为mindspore.Tensor时，
            通过此参数指定转换方法，采用False主要考虑以获得更好的性能（默认为True）。

    返回：
        TupleIterator，基于数据集对象创建的元组迭代器。

    示例：
        >>> # dataset是数据集类的实例化对象
        >>> iterator = dataset.create_tuple_iterator()
        >>> for item in iterator：
        ...     # item 是一个列表
        ...     print(type(item))
        ...     break
        <class 'list'>