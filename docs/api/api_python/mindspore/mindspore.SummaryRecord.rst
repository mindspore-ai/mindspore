mindspore.SummaryRecord
================================

.. py:class:: mindspore.SummaryRecord(log_dir, file_prefix='events', file_suffix='_MS', network=None, max_file_size=None, raise_exception=False, export_options=None)

    SummaryRecord用于记录summary数据和lineage数据。

    该方法将在一个指定的目录中创建summary文件和lineage文件，并将数据写入文件。

    它通过执行 `record` 方法将数据写入文件。除了通过 `summary算子 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.9/summary_record.html#方式二-结合summary算子和summarycollector自定义收集网络中的数据>`_ 记录网络的数据外，SummaryRecord还支持通过 `自定义回调函数和自定义训练循环 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.9/summary_record.html#方式三-自定义callback记录数据>`_ 记录数据。

    .. note::
        - 使用SummaryRecord时，需要将代码放置到 `if __name__ == "__main__"` 中运行。
        - 确保在最后关闭SummaryRecord，否则进程不会退出。请参阅下面的示例部分，了解如何用两种方式正确关闭SummaryRecord。
        - 每次训练只允许创建一个SummaryRecord实例，否则会导致数据写入异常。
        - SummaryRecord仅支持Linux系统。
        - 编译MindSpore时，设置 `-s on` 关闭维测功能后，SummaryRecord不可用。

    参数：
        - **log_dir** (str) - `log_dir` 是用来保存summary文件的目录。
        - **file_prefix** (str) - 文件的前缀。默认值：`events` 。
        - **file_suffix** (str) - 文件的后缀。默认值：`_MS` 。
        - **network** (Cell) - 表示用于保存计算图的网络。默认值：None。
        - **max_file_size** (int, 可选) - 可写入磁盘的每个文件的最大大小（以字节为单位）。例如，预期写入文件最大不超过4GB，则设置 `max_file_size=4*1024**3` 。默认值：None，表示无限制。
        - **raise_exception** (bool, 可选) - 设置在记录数据中发生RuntimeError或OSError异常时是否抛出异常。默认值：False，表示打印错误日志，不抛出异常。
        - **export_options** (Union[None, dict]) - 可以将保存在summary中的数据导出，并使用字典自定义所需的数据和文件格式。注：导出的文件大小不受 `max_file_size` 的限制。例如，您可以设置{'tensor_format':'npy'}将Tensor导出为 `npy` 文件。支持导出的数据类型如下所示。默认值：None，表示不导出数据。

          - **tensor_format** (Union[str, None]) - 自定义导出的Tensor的格式。支持["npy", None]。默认值：None，表示不导出Tensor。

            - **npy**：将Tensor导出为NPY文件。

    异常：
        - **TypeError** - `max_file_size` 不是整型，或 `file_prefix` 和 `file_suffix` 不是字符串。
        - **ValueError** - 编译MindSpore时，设置 `-s on` 关闭了维测功能。

    .. py:method:: add_value(plugin, name, value)

        添加需要记录的值。

        参数：
            - **plugin** (str) - 数据类型标签。

              - graph：代表添加的数据为计算图。
              - scalar：代表添加的数据为标量。
              - image：代表添加的数据为图片。
              - tensor：代表添加的数据为张量。
              - histogram：代表添加的数据为直方图。
              - train_lineage：代表添加的数据为训练阶段的lineage数据。
              - eval_lineage：代表添加的数据为评估阶段的lineage数据。
              - dataset_graph：代表添加的数据为数据图。
              - custom_lineage_data：代表添加的数据为自定义lineage数据。
              - LANDSCAPE：代表添加的数据为地形图。

            - **name** (str) - 数据名称。
            - **value** (Union[Tensor, GraphProto, TrainLineage, EvaluationLineage, DatasetGraph, UserDefinedInfo，LossLandscape]) - 待存储的值。

              - 当plugin为"graph"时，参数值的数据类型应为"GraphProto"对象。具体详情，请参见 `mindspore/ccsrc/anf_ir.proto <https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/ccsrc/utils/anf_ir.proto>`_ 。
              - 当plugin为"scalar"、"image"、"tensor"或"histogram"时，参数值的数据类型应为"Tensor"对象。
              - 当plugin为"train_lineage"时，参数值的数据类型应为"TrainLineage"对象。具体详情，请参见 `mindspore/ccsrc/lineage.proto <https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/ccsrc/utils/lineage.proto>`_ 。
              - 当plugin为"eval_lineage"时，参数值的数据类型应为"EvaluationLineage"对象。具体详情，请参见 `mindspore/ccsrc/lineage.proto <https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/ccsrc/utils/lineage.proto>`_ 。
              - 当plugin为"dataset_graph"时，参数值的数据类型应为"DatasetGraph"对象。具体详情，请参见 `mindspore/ccsrc/lineage.proto <https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/ccsrc/utils/lineage.proto>`_ 。
              - 当plugin为"custom_lineage_data"时，参数值的数据类型应为"UserDefinedInfo"对象。具体详情，请参见 `mindspore/ccsrc/lineage.proto <https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/ccsrc/utils/lineage.proto>`_ 。
              - 当plugin为"LANDSCAPE"时，参数值的数据类型应为"LossLandscape"对象。具体详情，请参见 `mindspore/ccsrc/summary.proto <https://gitee.com/mindspore/mindspore/blob/r1.10/mindspore/ccsrc/utils/summary.proto>`_ 。

        异常：
            - **ValueError** - `plugin` 的值不在可选值内。
            - **TypeError** - `name` 不是非空字符串，或当 `plugin` 为"scalar"、"image"、"tensor"或"histogram"时，`value` 的数据类型不是"Tensor"对象。

    .. py:method:: close()

        将缓冲区中的数据立刻写入文件并关闭SummaryRecord。请使用with语句或try…finally语句进行自动关闭。

    .. py:method:: flush()

        刷新缓冲区，将缓冲区中的数据写入磁盘。

        调用该函数以确保所有挂起事件都已写入到磁盘。

    .. py:method:: log_dir
        :property:

        获取日志文件的完整路径。

        返回：
            str，日志文件的完整路径。

    .. py:method:: record(step, train_network=None, plugin_filter=None)

        记录summary。

        参数：
            - **step** (int) - 表示当前的step。
            - **train_network** (Cell) - 表示用于保存计算图的训练网络。默认值：None，表示当原始网络的图为None时，不保存计算图。
            - **plugin_filter** (Callable[[str], bool], 可选) - 过滤器函数，用于过滤需要写入的标签项。默认值：None。

        返回：
            bool，表示记录是否成功。

        异常：
            - **TypeError** - `step` 不为整型，或 `train_network` 的类型不为 `mindspore.nn.Cell <https://www.mindspore.cn/docs/zh-CN/r1.10/api_python/nn/mindspore.nn.Cell.html?highlight=MindSpore.nn.cell#mindspore-nn-cell>`_ 。

    .. py:method:: set_mode(mode)

        设置模型运行阶段。不同的阶段会影响记录数据的内容。

        参数：
            - **mode** (str) - 待设置的网络阶段，可选值为"train"或"eval"。

              - train：代表训练阶段。
              - eval：代表评估阶段，此时 `summary_record` 不会记录summary算子的数据。

        异常：
            - **ValueError** - `mode` 的值不在可选值内。
