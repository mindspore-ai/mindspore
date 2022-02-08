.. py:class:: mindspore.train.callback.SummaryCollector(summary_dir, collect_freq=10, collect_specified_data=None, keep_default_action=True, custom_lineage_data=None, collect_tensor_freq=None, max_file_size=None, export_options=None)

    SummaryCollector可以收集一些常用信息。

    它可以帮助收集loss、学习率、计算图等。
    SummaryCollector还可以允许通过 `summary算子 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/summary_record.html#summarysummarycollector>`_ 将数据收集到summary文件中。

    .. note::
        - 不允许在回调列表中存在多个SummaryCollector实例。
        - 并非所有信息都可以在训练阶段或评估阶段收集。
        - SummaryCollector始终记录summary算子收集的数据。
        - SummaryCollector仅支持Linux系统。
        - 编译MindSpore时，设置 `-s on` 关闭维测功能后，SummaryCollector不可用。

    **参数：**

    - **summary_dir** (str) - 收集的数据将存储到此目录。如果目录不存在，将自动创建。
    - **collect_freq** (int) - 设置数据收集的频率，频率应大于零，单位为 `step` 。如果设置了频率，将在(current steps % freq)=0时收集数据，并且将总是收集第一个step。需要注意的是，如果使用数据下沉模式，单位将变成 `epoch` 。不建议过于频繁地收集数据，因为这可能会影响性能。默认值：10。
    - **collect_specified_data** (Union[None, dict]) - 对收集的数据进行自定义操作。您可以使用字典自定义需要收集的数据类型。例如，您可以设置{'collect_metric':False}不去收集metrics。支持控制的数据如下。默认值：None，收集所有数据。

      - **collect_metric** (bool) - 表示是否收集训练metrics，目前只收集loss。把第一个输出视为loss，并且算出其平均数。默认值：True。
      - **collect_graph** (bool) - 表示是否收集计算图。目前只收集训练计算图。默认值：True。
      - **collect_train_lineage** (bool) - 表示是否收集训练阶段的lineage数据，该字段将显示在MindInsight的 `lineage页面 <https://www.mindspore.cn/mindinsight/docs/zh-CN/master/lineage_and_scalars_comparison.html>`_ 上。默认值：True。
      - **collect_eval_lineage** (bool) - 表示是否收集评估阶段的lineage数据，该字段将显示在MindInsight的lineage页面上。默认值：True。
      - **collect_input_data** (bool) - 表示是否为每次训练收集数据集。目前仅支持图像数据。如果数据集中有多列数据，则第一列应为图像数据。默认值：True。
      - **collect_dataset_graph** (bool) - 表示是否收集训练阶段的数据集图。默认值：True。
      - **histogram_regular** (Union[str, None]) - 收集参数分布页面的权重和偏置，并在MindInsight中展示。此字段允许正则表达式控制要收集的参数。不建议一次收集太多参数，因为这会影响性能。注：如果收集的参数太多并且内存不足，训练将会失败。默认值：None，表示只收集网络的前五个超参。

    - **keep_default_action** (bool) - 此字段影响 `collect_specified_data` 字段的收集行为。True：表示设置指定数据后，其他数据按默认设置收集。False：表示设置指定数据后，只收集指定数据，不收集其他数据。默认值：True。
    - **custom_lineage_data** (Union[dict, None]) - 允许您自定义数据并将数据显示在MindInsight的lineage页面上。在自定义数据中，key支持str类型，value支持str、int和float类型。默认值：None，表示不存在自定义数据。
    - **collect_tensor_freq** (Optional[int]) - 语义与 `collect_freq` 的相同，但仅控制TensorSummary。由于TensorSummary数据太大，无法与其他summary数据进行比较，因此此参数用于降低收集量。默认情况下，收集TensorSummary数据的最大step数量为20，但不会超过收集其他summary数据的step数量。例如，给定 `collect_freq=10` ，当总step数量为600时，TensorSummary将收集20个step，而收集其他summary数据时会收集61个step。但当总step数量为20时，TensorSummary和其他summary将收集3个step。另外请注意，在并行模式下，会平均分配总的step数量，这会影响TensorSummary收集的step的数量。默认值：None，表示要遵循上述规则。
    - **max_file_size** (Optional[int]) - 可写入磁盘的每个文件的最大大小（以字节为单位）。例如，如果不大于4GB，则设置 `max_file_size=4*1024**3` 。默认值：None，表示无限制。
    - **export_options** (Union[None, dict]) - 表示对导出的数据执行自定义操作。注：导出的文件的大小不受 `max_file_size` 的限制。您可以使用字典自定义导出的数据。例如，您可以设置{'tensor_format':'npy'}将tensor导出为 `npy` 文件。支持控制的数据如下所示。默认值：None，表示不导出数据。

      - **tensor_format** (Union[str, None]) - 自定义导出的tensor的格式。支持["npy", None]。默认值：None，表示不导出tensor。

        - **npy** - 将tensor导出为NPY文件。

    **异常：**

    - **ValueError：** 编译MindSpore时，设置 `-s on` 关闭了维测功能。
