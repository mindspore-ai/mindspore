mindspore.ops.coo_add
================================

.. py:function:: mindspore.ops.coo_add(x1: COOTensor, x2: COOTensor, thresh: Tensor)

    两个COOTensor相加，根据相加的结果与 `thresh` 返回新的COOTensor。

    参数：
        - **x1** (COOTensor) - 一个操作数，与当前操作数相加。
        - **x2** (COOTensor) - 另一个操作数，与当前操作数相加。
        - **thresh** (Tensor) - 零维Tensor，其值作为阈值，用来决定相加后的结果Tensor中的哪些value/index对在最终返回值中出现。如果结果中value的数据类型为实数，则 `thresh` 的数据类型应该与它的数据类型一致。如果结果中value小于 `thresh`, 它将会被丢掉。

    返回：
        COOTensor，为两COOTensor相加后的结果。

    异常：
        - **ValueError** - 如果操作数(x1/x2)的indices的维度不等于2。
        - **ValueError** - 如果操作数(x1/x2)的values的维度不等于1。
        - **ValueError** - 如果操作数(x1/x2)的shape的维度不等于1。
        - **ValueError** - 如果thresh的维度不等于0。
        - **TypeError** - 如果操作数(x1/x2)的indices的数据类型不为int64。
        - **TypeError** - 如果操作数(x1/x2)的shape的数据类型不为int64。
        - **ValueError** - 如果操作数(x1/x2)的indices的长度不等于它的values的长度。
        - **TypeError** - 如果操作数(x1/x2)的values的数据类型不为(int8/int16/int32/int64/float32/float64/complex64/complex128)中的任何一个。
        - **TypeError** - 如果thresh的数据类型不为(int8/int16/int32/int64/float32/float64)中的任何一个。
        - **TypeError** - 如果操作数x1的indices数据类型不等于x2的indices数据类型。
        - **TypeError** - 如果操作数x1的values数据类型不等于x2的values数据类型。
        - **TypeError** - 如果操作数x1的shape数据类型不等于x2的shape数据类型。
        - **TypeError** - 如果操作数(x1/x2)的values的数据类型与thresh数据类型不匹配。
