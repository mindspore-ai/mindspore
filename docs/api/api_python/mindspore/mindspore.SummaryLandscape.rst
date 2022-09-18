mindspore.SummaryLandscape
================================

.. py:class:: mindspore.SummaryLandscape(summary_dir)

    SummaryLandscape可以帮助您收集loss地形图的信息。通过计算loss，可以在PCA（Principal Component Analysis）方向或者随机方向创建地形图。

    .. note::
        - 使用SummaryLandscape时，需要将代码放置到 `if __name__ == "__main__"` 中运行。
        - SummaryLandscape仅支持Linux系统。

    参数：
        - **summary_dir** (str) - 该路径将被用来保存创建地形图所使用的数据。


    .. py:method:: clean_ckpt()

        清理checkpoint。

    .. py:method:: gen_landscapes_with_multi_process(callback_fn, collect_landscape=None, device_ids=None, output=None)

        使用多进程来生成地形图。

        参数：
            - **callback_fn** (python function) - Python函数对象，用户需要写一个没有输入的函数，返回值要求如下。

              - mindspore.Model：用户的模型。
              - mindspore.nn.Cell：用户的网络。
              - mindspore.dataset：创建loss所需要的用户数据集。
              - mindspore.train.Metrics：用户的评估指标。

            - **collect_landscape** (Union[dict, None]) - 创建loss地形图所用的参数含义与SummaryCollector同名字段一致。此处设置的目的是允许用户可以自由修改创建loss地形图参数。默认值：None。

              - **landscape_size** (int) - 指定生成loss地形图的图像分辨率。例如：如果设置为128，则loss地形图的分辨率是128*128。计算loss地形图的时间随着分辨率的增大而增加。默认值：40。可选值：3-256。
              - **create_landscape** (dict) - 选择创建哪种类型的loss地形图，分为训练过程loss地形图（train）和训练结果loss地形图（result）。默认值：{"train": True, "result": True}。可选值：True/False。
              - **num_samples** (int) - 创建loss地形图所使用的数据集的大小。例如：在图像数据集中，您可以设置 `num_samples` 是128，这意味着将有128张图片被用来创建loss地形图。注意：`num_samples` 越大，计算loss地形图时间越长。默认值：128。
              - **intervals** (List[List[int]]) - 指定创建loss地形图所需要的checkpoint区间。例如：如果用户想要创建两张训练过程的loss地形图，分别为1-5epoch和6-10epoch，则用户可以设置[[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]]。注意：每个区间至少包含3个epoch。

            - **device_ids** (List(int)) - 指定创建loss地形图所使用的目标设备的ID。例如：[0, 1]表示使用设备0和设备1来创建loss地形图。默认值：None。
            - **output** (str) - 指定保存loss地形图的路径。默认值：None。默认保存路径与summary文件相同。
