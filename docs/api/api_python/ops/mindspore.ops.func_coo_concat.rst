mindspore.ops.coo_concat
===========================

.. py:function:: mindspore.ops.coo_concat(sp_input, concat_dim=0)

    根据指定的轴concat_dim对输入的COO Tensor（sp_input）进行合并操作。

    .. warning::
        这是一个实验性API，后续可能修改或删除。目前只支持CPU。

    参数：
        - **sp_input** (Union[list(COOTensor), tuple(COOTensor)]) - 输入的需要concat合并的稀疏张量。
        - **concat_dim** (scalar) - 指定需要合并的轴序号, 它的取值必须是在[-rank, rank)之内，
          其中rank为sp_input中COOTensor的shape的维度值，默认值： ``0`` 。

    返回：
        COOTensor，按concat_dim轴合并后的COOTensor。这个COOTensor的稠密shape值为：
        非concat_dim轴shape与输入一致，concat_dim轴shape是所有输入对应轴shape的累加。

    异常：
        - **ValueError** - 如果只有一个COOTensor输入，报错。
        - **ValueError** - 如果输入的COOTensor的shape维度大于3，COOTensor的构造会报错。目前COOTensor的shape维度只能为2。
