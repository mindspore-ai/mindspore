mindspore.ops.ParallelConcat
=============================

.. py:class:: mindspore.ops.ParallelConcat

    根据第一个维度连接Tensor。

    Concat和并行Concat之间的区别在于，Concat要求在操作开始之前需要计算所有的输入，但不要求在构图期间知道输入shape。
    ParallelConcat将在输入片段可用时，将其复制到输出中，在某些情况下，这可以提供性能优势。

    .. note::
        输入Tensor在第一个维度要求长度为1。
        
    输入：
        - **values** (tuple, list) - 由Tensor组成的tuple或list。其元素的数据类型和shape必须相同。数据类型为数值型，但float64除外。
        
    输出：
        Tensor，数据类型与 `values` 相同。
        
    异常：
        - **ValueError** - 如果 `values` 的shape长度小于1。
        - **ValueError** - 如果 `values` 中各个Tensor的数据类型和shape不相同。
