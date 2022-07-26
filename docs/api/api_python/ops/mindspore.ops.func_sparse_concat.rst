mindspore.ops.sparse_concat
===========================

.. py:function:: mindspore.ops.sparse_concat(sp_input, concat_dim)

    根据指定的轴concat_dim对输入的COO Tensor（sp_input）进行合并操作。

    .. note::
        实验特性接口，目前只支持CPU。

    参数：
        - **sp_input** (Union[list(COOTnesor), tuple(COOTensor)) - 输入的需要concat合并的稀疏张量。
        - **concat_dim** (标量) - 指定需要合并的轴序号，它的取值必须是在[-rank, rank)之内，
            其中rank为sp_input中COOTensor的shape的纬度值

    返回：
        COOTensor，按concat_dim轴合并后的COOTensor。

    异常：
        - **ValueError** - 如果只有一个COOTensor输入，报错。
        - **ValueError** - 如果输入的COOTensor的shape纬度大于3。COOTensor的构造会报错，
            目前COOTensor的shape纬度只能为2
