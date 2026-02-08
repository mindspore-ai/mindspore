mindspore.dataset.RandomSampler
================================

.. py:class:: mindspore.dataset.RandomSampler(replacement=False, num_samples=None, shuffle=Shuffle.GLOBAL)

    随机采样器。

    .. note::
        不同数据集的混洗模式支持如下：

        .. list-table:: 混洗模式支持
            :widths: 50 50 50 50
            :header-rows: 1

            * - 混洗模式
              - MindDataset
              - TFRecordDataset
              - 其他数据集
            * - ``Shuffle.ADAPTIVE``
              - 支持
              - 不支持
              - 不支持
            * - ``Shuffle.GLOBAL``
              - 支持
              - 支持
              - 支持
            * - ``Shuffle.PARTIAL``
              - 支持
              - 不支持
              - 不支持
            * - ``Shuffle.FILES``
              - 支持
              - 支持
              - 不支持
            * - ``Shuffle.INFILE``
              - 支持
              - 不支持
              - 不支持

    参数：
        - **replacement** (bool, 可选) - 是否将样本ID放回，用于下一次采样。默认值： ``False`` ，无放回采样。
        - **num_samples** (int, 可选) - 获取的样本数，可用于部分获取采样得到的样本。默认值： ``None`` ，获取采样到的所有样本。
        - **shuffle** (Shuffle, 可选) - 采用何种混洗逻辑。默认值： ``Shuffle.GLOBAL`` ，全局混洗。
          通过传入枚举变量设置数据混洗的模式，枚举变量参考链接 :class:`mindspore.dataset.Shuffle` ：

          - ``Shuffle.ADAPTIVE`` ：当数据集样本小于等于1亿时，采用 ``Shuffle.GLOBAL`` ；当大于1亿时，采用局部 ``Shuffle.PARTIAL`` ，每100万样本混洗一次。
          - ``Shuffle.GLOBAL`` ：执行全局混洗，一次性混洗数据集中所有样本。占用内存大。
          - ``Shuffle.PARTIAL`` ：执行局部混洗，每100万个样本混洗一次。占用内存小于 ``Shuffle.GLOBAL`` 。
          - ``Shuffle.FILES`` ：仅混洗文件序列，不混洗文件中的数据。
          - ``Shuffle.INFILE`` ：保持读入文件的序列，仅混洗每个文件中的数据。

    异常：
        - **TypeError** - `replacement` 不是bool值。
        - **TypeError** - `num_samples` 不是整数值。
        - **ValueError** - `num_samples` 为负值。
        - **TypeError** - `shuffle` 的类型不是Shuffle。

    .. include:: mindspore.dataset.BuiltinSampler.rst

    .. include:: mindspore.dataset.BuiltinSampler.b.rst
