mindspore.dataset.Shuffle
=========================

.. py:class:: mindspore.dataset.Shuffle

    指定混洗模式的枚举类。

    - **Shuffle.FALSE** - 关闭数据集混洗。
    - **Shuffle.ADAPTIVE** - 当数据集样本小于等于1亿时，采用 ``Shuffle.GLOBAL`` ；当大于1亿时，采用局部 ``Shuffle.PARTIAL`` ，每100万样本混洗一次。
    - **Shuffle.GLOBAL** - 执行全局混洗，一次性混洗数据集中所有样本。
    - **Shuffle.PARTIAL** - 执行局部混洗，每100万个样本混洗一次。
    - **Shuffle.FILES** - 仅混洗文件，不混洗文件中的数据。
    - **Shuffle.INFILE** - 保持读入文件的序列，仅混洗每个文件中的数据。
