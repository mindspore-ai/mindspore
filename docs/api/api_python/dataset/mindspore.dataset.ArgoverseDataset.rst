mindspore.dataset.ArgoverseDataset
====================================

.. py:class:: mindspore.dataset.ArgoverseDataset(data_dir, column_names="graph", shuffle=None, num_parallel_workers=1, python_multiprocessing=True, perf_mode=True)

    加载argoverse数据集并进行图（Graph）初始化。

    Argoverse数据集是自动驾驶领域的公共数据集，当前实现的 `ArgoverseDataset` 主要用于加载argoverse数据集中运动预测（Motion Forecasting）场景的数据集，具体信息可访问官网了解：
    https://www.argoverse.org/av1.html#download-link。

    参数：
        - **data_dir** (str) - 加载数据集的目录，这里包含原始格式的数据，并将在 `process` 方法中被加载。
        - **column_names** (Union[str, list[str]]，可选) - dataset包含的单个列名或多个列名组成的列表，默认值：'Graph'。当实现类似 `__getitem__` 等方法时，列名的数量应该等于该方法中返回数据的条数，如下述示例，建议初始化时明确它的取值如：`column_names=["edge_index", "x", "y", "cluster", "valid_len", "time_step_len"]`。
        - **num_parallel_workers** (int, 可选) - 指定读取数据的工作进程数/线程数（由参数 `python_multiprocessing` 决定当前为多进程模式或多线程模式），默认值：1。
        - **shuffle** (bool，可选) - 是否混洗数据集。当实现的Dataset带有可随机访问属性（ `__getitem__` ）时，才可以指定该参数。默认值：None。
        - **python_multiprocessing** (bool，可选) - 启用Python多进程模式加速运算，默认值：True。当传入 `source` 的Python对象的计算量很大时，开启此选项可能会有较好效果。
        - **perf_mode** (bool，可选) - 遍历创建的dataset对象时获得更高性能的模式（在此过程中将调用 `__getitem__` 方法）。默认值：True，将Graph的所有数据（如边的索引、节点特征和图的特征）都作为图特征进行存储。


    .. py:method:: load()

        从给定（处理好的）路径加载数据，也可以在自己实现的Dataset类中实现这个方法。

    .. py:method:: process()

        针对argoverse数据集的处理方法，基于加载上来的原始数据集创建很多子图。
        数据预处理方法主要参考：https://github.com/xk-huang/yet-another-vectornet/blob/master/dataset.py。

    .. py:method:: save()

        将经过 `process` 函数处理后的数据以 numpy.npz 格式保存到磁盘中，也可以在自己实现的Dataset类中自己实现这个方法。

.. include:: mindspore.dataset.api_list_vision.rst
