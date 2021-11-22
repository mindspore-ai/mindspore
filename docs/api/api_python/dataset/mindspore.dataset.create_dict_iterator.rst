create_dict_iterator(num_epochs=-1, output_numpy=False)

        基于数据集对象创建迭代器，输出数据为字典类型。

        字典中列的顺序可能与数据集对象中原始顺序不同。

        参数：
            num_epochs (int, optional)：迭代器可以迭代的最多轮次数（默认为-1，迭代器可以迭代无限次）。
            output_numpy (bool, optional)：是否输出NumPy数据类型，如果`output_numpy`为False，
            迭代器输出的每列数据类型为MindSpore.Tensor（默认为False）。

        返回：
            DictIterator，基于数据集对象创建的字典迭代器。

        示例：
            >>> # dataset是数据集类的实例化对象
            >>> iterator = dataset.create_dict_iterator()
            >>> for item in iterator:
            ...     # item 是一个dict
            ...     print(type(item))
            ...     break
            <class 'dict'>