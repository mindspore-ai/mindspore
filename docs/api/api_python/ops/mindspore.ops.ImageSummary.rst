mindspore.ops.ImageSummary
==========================

.. py:class:: mindspore.ops.ImageSummary

    将图像保存到Summary文件。必须和SummaryRecord或SummaryCollector一起使用，
    Summary文件的保存路径由SummaryRecord或SummaryCollector指定。Summary文件可以通过MindInsight加载并展示，
    关于MindInsight的详细信息请参考 `MindInsight文档 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0.0-alpha/index.html>`_ 。

    输入：
        - **name** (str) - 输入变量的名称，不能是空字符串。
        - **value** (Tensor) - 图像数据的值，Tensor的rank必须为4。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
