mindspore.dataset.Dataset.filter
================================

.. py:method:: mindspore.dataset.Dataset.filter(predicate, input_columns=None, num_parallel_workers=None)

    通过自定义判断条件对数据集对象中的数据进行过滤。

    参数：
        - **predicate** (callable) - Python可调用对象。要求该对象接收n个入参，用于指代每个数据列的数据，最后返回值一个bool值。
          如果返回值为False，则表示过滤掉该条数据。注意n的值与参数 `input_columns` 表示的输入列数量一致。
        - **input_columns** (Union[str, list[str]], 可选) - `filter` 操作的输入数据列。默认值：None，`predicate` 将应用于数据集中的所有列。
        - **num_parallel_workers** (int, 可选) - 指定 `filter` 操作的并发线程数。默认值：None，使用全局默认线程数(8)，也可以通过 `mindspore.dataset.config.set_num_parallel_workers` 配置全局线程数。

    返回：
        Dataset，执行给定筛选过滤操作的数据集对象。
