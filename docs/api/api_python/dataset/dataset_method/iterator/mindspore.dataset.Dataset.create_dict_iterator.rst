mindspore.dataset.Dataset.create_dict_iterator
==============================================

.. py:method:: mindspore.dataset.Dataset.create_dict_iterator(num_epochs=-1, output_numpy=False)

    基于数据集对象创建迭代器。输出的数据为字典类型。

    参数：
        - **num_epochs** (int, 可选) - 迭代器可以迭代的最大次数。默认值：-1，迭代器可以迭代无限次。
        - **output_numpy** (bool, 可选) - 输出的数据是否转为NumPy类型。如果为False，迭代器输出的每列数据类型为MindSpore.Tensor，否则为NumPy。默认值：False。

    返回：
        DictIterator，基于数据集对象创建的字典迭代器。
