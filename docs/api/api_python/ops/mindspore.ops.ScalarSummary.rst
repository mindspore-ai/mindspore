mindspore.ops.ScalarSummary
============================

.. py:class:: mindspore.ops.ScalarSummary

    将标量数据保存到Summary文件。必须和SummaryRecord或SummaryCollector一起使用，
    Summary文件的保存路径由SummaryRecord或SummaryCollector指定。Summary文件可以通过MindInsight加载并展示，
    关于MindInsight的详细信息请参考 `MindInsight文档 <https://www.mindspore.cn/mindinsight/docs/zh-CN/r2.0.0-alpha/index.html>`_ 。

    输入：
        - **name** (str) - 输入标量的名称，不能是空字符串。
        - **value** (Tensor) - 标量数据的值，维度必须为0或者1。

    异常：
        - **TypeError** - 如果 `name` 不是str。
        - **TypeError** - 如果 `value` 不是Tensor。
        - **ValueError** - 如果 `value` 的维度大于1。
