mindspore.SummaryCollector
================================

.. py:class:: mindspore.SummaryCollector(summary_dir, collect_freq=10, collect_specified_data=None, keep_default_action=True, custom_lineage_data=None, collect_tensor_freq=None, max_file_size=None, export_options=None)

    SummaryCollector可以收集一些常用信息。

    它可以帮助收集loss、学习率、计算图等。
    SummaryCollector还可以允许通过 `summary算子 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.9/summary_record.html#方式二-结合summary算子和summarycollector自定义收集网络中的数据>`_ 将数据收集到summary文件中。

    .. note::
        - 使用SummaryCollector时，需要将代码放置到 `if __name__ == "__main__"` 中运行。
        - 不允许在回调列表中存在多个SummaryCollector实例。
        - 并非所有信息都可以在训练阶段或评估阶段收集。
        - SummaryCollector始终记录summary算子收集的数据。
        - SummaryCollector仅支持Linux系统。
        - 编译MindSpore时，设置 `-s on` 关闭维测功能后，SummaryCollector不可用。

    参数：
        - **summary_dir** (str) - 收集的数据将存储到此目录。如果目录不存在，将自动创建。
        - **collect_freq** (int) - 设置数据收集的频率，频率应大于零，单位为 `step` 。如果设置了频率，将在(current steps % freq)=0时收集数据，并且将总是收集第一个step。需要注意的是，如果使用数据下沉模式，单位将变成 `epoch` 。不建议过于频繁地收集数据，因为这可能会影响性能。默认值：10。
        - **collect_specified_data** (Union[None, dict]) - 对收集的数据进行自定义操作。您可以使用字典自定义需要收集的数据类型。例如，您可以设置{'collect_metric':False}不去收集metrics。支持控制的数据如下。默认值：None，收集所有数据。

          - **collect_metric** (bool) - 表示是否收集训练metrics，目前只收集loss。把第一个输出视为loss，并且算出其平均数。默认值：True。
          - **collect_graph** (bool) - 表示是否收集计算图。目前只收集训练计算图。默认值：True。
          - **collect_train_lineage** (bool) - 表示是否收集训练阶段的lineage数据，该字段将显示在MindInsight的 `lineage页面 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r1.9/lineage_and_scalars_comparison.html>`_ 上。默认值：True。
          - **collect_eval_lineage** (bool) - 表示是否收集评估阶段的lineage数据，该字段将显示在MindInsight的lineage页面上。默认值：True。
          - **collect_input_data** (bool) - 表示是否为每次训练收集数据集。目前仅支持图像数据。如果数据集中有多列数据，则第一列应为图像数据。默认值：True。
          - **collect_dataset_graph** (bool) - 表示是否收集训练阶段的数据集图。默认值：True。
          - **histogram_regular** (Union[str, None]) - 收集参数分布页面的权重和偏置，并在MindInsight中展示。此字段允许正则表达式控制要收集的参数。不建议一次收集太多参数，因为这会影响性能。注：如果收集的参数太多并且内存不足，训练将会失败。默认值：None，表示只收集网络的前五个超参。
          - **collect_landscape** (Union[dict, None]) - 表示是否收集创建loss地形图所需要的参数。如果设置None，则不收集任何参数。默认收集所有参数并且将会保存在 `{summary_dir}/ckpt_dir/train_metadata.json` 文件中。

            - **landscape_size** (int) - 指定生成loss地形图的图像分辨率。例如：如果设置为128，则loss地形图的分辨率是128*128。注意：计算loss地形图的时间随着分辨率的增大而增加。默认值：40。可选值：3-256。
            - **unit** (str) - 指定训练过程中保存checkpoint时，下方参数 `intervals` 以何种形式收集模型权重。例如：将 `intervals` 设置为[[1, 2, 3, 4]]，如果 `unit` 设置为step，则收集模型权重的频率单位为step，将保存1-4个step的模型权重，而 `unit` 设置为epoch，则将保存1-4个epoch的模型权重。默认值：step。可选值：epoch/step。
            - **create_landscape** (dict) - 选择创建哪种类型的loss地形图，分为训练过程loss地形图（train）和训练结果loss地形图（result）。默认值：{"train": True, "result": True}。可选值：True/False。
            - **num_samples** (int) - 创建loss地形图所使用的数据集的大小。例如：在图像数据集中，您可以设置 `num_samples` 是128，这意味着将有128张图片被用来创建loss地形图。注意：`num_samples` 越大，计算loss地形图时间越长。默认值：128。
            - **intervals** (List[List[int]]) - 指定loss地形图的区间。例如：如果用户想要创建两张训练过程的loss地形图，分别为1-5epoch和6-10epoch，则用户可以设置[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]。注意：每个区间至少包含3个epoch。

        - **keep_default_action** (bool) - 此字段影响 `collect_specified_data` 字段的收集行为。True：表示设置指定数据后，其他数据按默认设置收集。False：表示设置指定数据后，只收集指定数据，不收集其他数据。默认值：True。
        - **custom_lineage_data** (Union[dict, None]) - 允许您自定义数据并将数据显示在MindInsight的lineage页面上。在自定义数据中，key支持str类型，value支持str、int和float类型。默认值：None，表示不存在自定义数据。
        - **collect_tensor_freq** (Optional[int]) - 语义与 `collect_freq` 的相同，但仅控制TensorSummary。由于TensorSummary数据太大，无法与其他summary数据进行比较，因此此参数用于降低收集量。默认情况下，收集TensorSummary数据的最大step数量为20，但不会超过收集其他summary数据的step数量。例如，给定 `collect_freq=10` ，当总step数量为600时，TensorSummary将收集20个step，而收集其他summary数据时会收集61个step。但当总step数量为20时，TensorSummary和其他summary将收集3个step。另外请注意，在并行模式下，会平均分配总的step数量，这会影响TensorSummary收集的step的数量。默认值：None，表示要遵循上述规则。
        - **max_file_size** (Optional[int]) - 可写入磁盘的每个文件的最大大小（以字节为单位）。例如，如果不大于4GB，则设置 `max_file_size=4*1024**3` 。默认值：None，表示无限制。
        - **export_options** (Union[None, dict]) - 表示对导出的数据执行自定义操作。注：导出的文件的大小不受 `max_file_size` 的限制。您可以使用字典自定义导出的数据。例如，您可以设置{'tensor_format':'npy'}将tensor导出为 `npy` 文件。支持控制的数据如下所示。默认值：None，表示不导出数据。

          - **tensor_format** (Union[str, None]) - 自定义导出的tensor的格式。支持["npy", None]。默认值：None，表示不导出tensor。

            - **npy** - 将tensor导出为NPY文件。

    异常：
        - **ValueError** - 译MindSpore时，设置 `-s on` 关闭了维测功能。
