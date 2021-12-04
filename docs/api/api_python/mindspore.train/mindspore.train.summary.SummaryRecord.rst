.. py:class:: mindspore.train.summary.SummaryRecord(log_dir, file_prefix='events', file_suffix='_MS', network=None, max_file_size=None, raise_exception=False, export_options=None)

    SummaryRecord用于记录summary数据和lineage数据。

    该API将在一个指定的目录中创建summary文件和lineage文件，并将数据写入文件。

    它通过执行 `record` 方法将数据写入文件。除了通过定义summary算子记录从网络获取的数据外，SummaryRecord还支持记录其他数据，这些数据可以通过调用 `add_value` 添加。

    .. note::
        - 确保在最后关闭SummaryRecord，否则进程不会退出。请参阅下面的示例部分，了解如何用两种方式正确关闭SummaryRecord。
        - 每次只允许一个SummaryRecord实例，否则会导致数据写入异常。
        - SummaryRecord仅支持Linux系统。

    **参数：**

    - **log_dir** (str) - `log_dir` 是用来保存summary的目录。
    - **file_prefix** (str) - 文件的前缀。默认值：events。
    - **file_suffix** (str) - 文件的后缀。默认值：_MS。
    - **network** (Cell) - 通过网络获取用于保存图形summary的管道。默认值：None。
    - **max_file_size** (int, optional) - 可写入磁盘的每个文件的最大大小（以字节为单位）。例如，如果不大于4GB，则设置 `max_file_size=4*1024**3` 。默认值：None，表示无限制。
    - **raise_exception** (bool, 可选) - 设置在记录数据中发生RuntimeError或OSError异常时是否抛出异常。默认值：False，表示打印错误日志，不抛出异常。
    - **export_options** (Union[None, dict]) - 可以将保存在summary中的数据导出，并使用字典自定义所需的数据和文件格式。注：导出的文件大小不受 `max_file_size` 的限制。例如，您可以设置{'tensor_format':'npy'}将Tensor导出为NPY文件。支持控制的数据如下所示。默认值：None，表示不导出数据。

        - **tensor_format** (Union[str, None]) - 自定义导出的Tensor的格式。支持["npy", None]。默认值：None，表示不导出Tensor。

          - **npy**：将Tensor导出为NPY文件。

    **异常：**

    - **TypeError：** 参数类型不正确。
    - **RuntimeError** ：运行时错误。
    - **OSError：** 系统错误。

    **样例：**

    >>> from mindspore.train.summary import SummaryRecord
    >>> if __name__ == '__main__':
    ...     # 在with语句中使用以自动关闭
    ...     with SummaryRecord(log_dir="./summary_dir") as summary_record:
    ...         pass
    ...
    ...     # 在try .. finally .. 语句中使用以确保关闭
    ...     try:
    ...         summary_record = SummaryRecord(log_dir="./summary_dir")
    ...     finally:
    ...         summary_record.close()

    .. py:method:: add_value(plugin, name, value)

        添加稍后记录的值。

        **参数：**

        - **plugin** (str) - 数据类型标签。
        - **name** (str) - 数据名称。
        - **value** (Union[Tensor, GraphProto, TrainLineage, EvaluationLineage, DatasetGraph, UserDefinedInfo])： 待存储的值。

          - 当plugin为"graph"时，参数值的数据类型应为"GraphProto"对象。具体详情，请参见 mindspore/ccsrc/anf_ir.proto。
          - 当plugin为"scalar"、"image"、"tensor"或"histogram"时，参数值的数据类型应为"Tensor"对象。
          - 当plugin为"train_lineage"时，参数值的数据类型应为"TrainLineage"对象。具体详情，请参见 mindspore/ccsrc/lineage.proto。
          - 当plugin为"eval_lineage"时，参数值的数据类型应为"EvaluationLineage"对象。具体详情，请参见 mindspore/ccsrc/lineage.proto。
          - 当plugin为"dataset_graph"时，参数值的数据类型应为"DatasetGraph"对象。具体详情，请参见 mindspore/ccsrc/lineage.proto。
          - 当plugin为"custom_lineage_data"时，参数值的数据类型应为"UserDefinedInfo"对象。具体详情，请参见 mindspore/ccsrc/lineage.proto。
          - 当plugin为"explainer"时，参数值的数据类型应为"Explain"对象。具体详情，请参见 mindspore/ccsrc/summary.proto。

        **异常：**

        - **ValueError：** 参数值无效。
        - **TypeError：** 参数类型错误。

        **样例：**

        >>> from mindspore import Tensor
        >>> from mindspore.train.summary import SummaryRecord
        >>> if __name__ == '__main__':
        ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
        ...         summary_record.add_value('scalar', 'loss', Tensor(0.1))

    .. py:method:: close()

        将所有事件持久化并关闭SummaryRecord。请使用with语句或try…finally语句进行自动关闭。

        **样例：**

        >>> from mindspore.train.summary import SummaryRecord
        >>> if __name__ == '__main__':
        ...     try:
        ...         summary_record = SummaryRecord(log_dir="./summary_dir")
        ...     finally:
        ...         summary_record.close()

    .. py:method:: flush()

        将事件文件持久化到磁盘。

        调用该函数以确保所有挂起事件都已写入到磁盘。

        **样例：**

        >>> from mindspore.train.summary import SummaryRecord
        >>> if __name__ == '__main__':
        ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
        ...         summary_record.flush()

    .. py:method:: log_dir
        :property:

        获取日志文件的完整路径。

        **返回：**

        str，日志文件的完整路径。

        **样例：**

        >>> from mindspore.train.summary import SummaryRecord
        >>> if __name__ == '__main__':
        ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
        ...         log_dir = summary_record.log_dir

    .. py:method:: record(step, train_network=None, plugin_filter=None)

        记录summary。

        **参数：**

        - **step** (int) - 表示训练step的编号。
        - **train_network** (Cell) - 表示用于保存图形的备用网络。默认值：None，表示当原始网络图为None时，不保存图形summary。
        - **plugin_filter** (Optional[Callable[[str], bool]]) - 过滤器函数，用于通过返回False来过滤正在写入的插件。默认值：None。

        **返回：**

        bool，表示记录进程是否成功。

        **异常：**

        - **TypeError：** 参数类型错误。
        - **RuntimeError：** 磁盘空间不足。

        **样例：**

        >>> from mindspore.train.summary import SummaryRecord
        >>> if __name__ == '__main__':
        ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
        ...         summary_record.record(step=2)
        ...
        True

    .. py:method:: set_mode(mode)

        设置训练阶段。不同的训练阶段会影响数据记录。

        **参数：**

        - **mode** (str) - 待设置的模式，为"train"或"eval"。当模式为"eval"时，`summary_record` 不记录summary算子的数据。

        **异常：**

        **ValueError：** 无法识别模式。

        **样例：**

        >>> from mindspore.train.summary import SummaryRecord
        >>> if __name__ == '__main__':
        ...     with SummaryRecord(log_dir="./summary_dir", file_prefix="xx_", file_suffix="_yy") as summary_record:
        ...         summary_record.set_mode('eval')