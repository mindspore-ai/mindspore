mindspore.mint.multinomial
==========================

.. py:function:: mindspore.mint.multinomial(input, num_samples, replacement=False, *, generator=None)

    根据输入生成一个多项式分布的Tensor。

    多项式分布是一种概率分布，把二项分布公式推广至多种状态，就得到了多项式分布。在多项式分布中，每个事件都有一个固定的概率，这些概率的和为1。

    :func:`mindspore.mint.multinomial` 接口的作用是对输入 `input` 进行 `num_samples` 次抽样，输出的Tensor则为每一次抽样时输入Tensor的索引，其中 `input` 中的值为每次抽样取到对应索引的概率。

    这里我们给一个相对极端的用例方便理解，我们给定一个输入概率值Tensor，值为 `Tensor([90 / 100, 10 / 100, 0], mindspore.float32)` ，代表我们一共可以对三个索引进行抽样，分别为索引0，索引1，索引2，它们被抽中的概率分别为90%，10%，0%，我们对其进行n次抽样，抽样的结果序列则为多项式分布的计算结果，计算结果长度与抽样次数一致。

    在样例代码case 1中，我们对其进行两次不放回抽样（`replacement` 为 ``False``），那我们的计算结果则大概率为 `[0, 1]` ，小概率为 `[1, 0]`， 由于每次抽样抽到索引0的概率为90%，因此抽到的结果序列中，第一次大概率是抽到索引0，由于抽到索引2的概率为0，因此抽样两次结果不可能出现索引2，那第二次结果一定是索引1，因此结果序列为 `[0, 1]`。

    在样例代码case 2中，我们对其进行10次放回抽样（`replacement` 为 ``True``），可以看到计算结果中大概有90%的抽样结果为抽到索引0，符合预期。

    在样例代码case 3中，我们将输入扩展为2维，可以看到抽样结果在每一个维度中结果也符合我们抽样预期。

    .. note::
        输入的行不需要求和为1（当使用值作为权重的情况下），但必须是非负的、有限的，并且和不能为0。在使用值作为权重的情况下，可以理解为对输入沿最后一维进行了归一化操作，以此保证概率和为1。

    .. warning::
        这是一个实验性API，后续可能修改或删除。

    参数：
        - **input** (Tensor) - 输入的概率值Tensor，必须是一维或二维。
        - **num_samples** (int) - 采样的次数。
        - **replacement** (bool, 可选) - 是否是可放回的采样，默认值： ``False`` 。

    关键字参数：
        - **generator** (generator，可选) - MindSpore随机种子。默认值： ``None``。

    返回：
        Tensor，数据类型为int64。
        如果 `input` 是向量，输出数据shape是 `num_samples` 的向量。
        如果 `input` 是 m 行矩阵，输出数据shape是 m * num_samples 大小的矩阵。

    异常：
        - **TypeError** - 如果 `num_samples` 不是int。
        - **RuntimeError** - :math:`\text{num_samples} <= 0`。
        - **RuntimeError** - `replacement` 为 ``False`` 时， :math:`\text{num_samples}` 大于 `input` 最后一维的shape。
        - **RuntimeError** - `input` 最后一维的shape超过 ``2^24``。
