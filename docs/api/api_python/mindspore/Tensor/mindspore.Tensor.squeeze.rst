mindspore.Tensor.squeeze
========================

.. py:method:: mindspore.Tensor.squeeze(axis=None)

    从Tensor中删除shape为1的维度。

    参数：
        - **axis** (Union[None, int, list(int), tuple(int)], 可选) - 选择shape中长度为1的条目的子集。如果选择shape条目长度大于1的轴，则报错。默认值为None。

    返回：
        Tensor，删除了长度为1的维度的全部子集或一个子集。

    异常：
        - **TypeError** - 输入的参数类型有误。
        - **ValueError** - 指定维度的shape大于1。