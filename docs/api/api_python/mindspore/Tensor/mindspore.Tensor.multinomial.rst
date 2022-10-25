mindspore.Tensor.multinomial
=============================

.. py:method:: mindspore.Tensor.multinomial(self, num_samples, seed=0, seed2=0):

    返回从相应的张量输入行。输入行不需要求和为1(在这种情况下，我们使用这些值作为权重)，但必须是非负的、有限的并且具有非零和。self必须是输入张量包含概率总和的，必须是1或2维。

    参数：
        - **num_samples** (int32)—要绘制的样本数
        - **seed** (int)：随机种子，必须为非负数。默认值：0
        - **seed2** (int)：随机seed2，必须为非负数。默认值：0

    输出：
        与self具有相同行的张量，每行具有num_samples采样索引
    异常：
        TypeError:如果`seed`和`seed2`都不是int
        TypeError:如果`self`不是数据类型为float32的Tensor
        TypeError:如果`num_samples`的数据类型不是int32
    支持的平台：
        ``GPU``
