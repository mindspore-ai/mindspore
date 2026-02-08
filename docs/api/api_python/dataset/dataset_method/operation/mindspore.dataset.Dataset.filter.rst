mindspore.dataset.Dataset.filter
================================

.. py:method:: mindspore.dataset.Dataset.filter(predicate, input_columns=None, num_parallel_workers=None)

    通过自定义判断条件对数据集对象中的数据进行过滤。

    参数：
        - **predicate** (callable) - Python可调用对象。要求该对象接收的入参数量与 `input_columns` 指定的输入列数量一致，每个入参对应一个数据列的数据，最后返回一个bool值。
          如果返回值为 ``False`` ，则表示过滤掉该条数据。
        - **input_columns** (Union[str, list[str]], 可选) - `filter` 操作的输入数据列。默认值： ``None`` ，`predicate` 将应用于数据集中的所有列。
        - **num_parallel_workers** (int, 可选) - 指定 `filter` 操作的并发线程数。默认值： ``None`` ，使用全局默认线程数（8），也可以通过 :func:`mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。

    返回：
        Dataset，应用了上述操作的新数据集对象。
