mindspore.ops.ReduceScatter
===========================

.. py:class:: mindspore.ops.ReduceScatter(op=ReduceOp.SUM, group=GlobalComm.WORLD_COMM_GROUP)

    规约并且分发指定通信组中的张量。

    .. note::
        在集合的所有过程中，Tensor必须具有相同的shape和格式。

    参数：
        - **op** (str, 可选) - 指定用于元素的规约操作，如SUM和MAX。默认值： ``ReduceOp.SUM`` 。
        - **group** (str, 可选) - 要处理的通信组。默认值： ``GlobalComm.WORLD_COMM_group`` 。

    输入：
        - **input_x** (Tensor) - 输入Tensor，假设其形状为 :math:`(N, *)` ，其中 `*` 为任意数量的额外维度。N必须能够被rank_size整除，rank_size为当前通讯组里面的计算卡数量。

    输出：
        Tensor，数据类型与 `input_x` 一致，shape为 :math:`(N/rank\_size, *)` 。

    异常：
        - **TypeError** - 如果 `op` 和 `group` 不是字符串。
        - **ValueError** - 如果输入的第一个维度不能被rank size整除。

    样例：

    .. note::
        .. include:: mindspore.ops.comm_note.rst
        
        该样例需要在2卡环境下运行。
    
    教程样例：
        - `分布式集合通信原语 - ReduceScatter
          <https://www.mindspore.cn/docs/zh-CN/master/api_python/samples/ops/communicate_ops.html#reducescatter>`_
