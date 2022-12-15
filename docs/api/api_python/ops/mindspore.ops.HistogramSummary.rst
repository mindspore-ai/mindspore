mindspore.ops.HistogramSummary
===============================

.. py:class:: mindspore.ops.HistogramSummary

    计算Tensor的直方图并保存到Summary文件。必须和SummaryRecord或SummaryCollector一起使用，
    Summary文件的保存路径由SummaryRecord或SummaryCollector指定。Summary文件可以通过MindInsight加载并展示，
    关于MindInsight的详细信息请参考 `MindInsight文档 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0.0-alpha/index.html>`_ 。

    输入：
        - **name** (str) - 输入变量的名称。
        - **value** (Tensor) - Tensor的值，Tensor的rank必须大于0。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
