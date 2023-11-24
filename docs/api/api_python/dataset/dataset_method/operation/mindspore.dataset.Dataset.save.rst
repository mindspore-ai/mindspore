mindspore.dataset.Dataset.save
===============================

.. py:method:: mindspore.dataset.Dataset.save(file_name, num_files=1, file_type='mindrecord')

    将数据处理管道中正处理的数据保存为通用的数据集格式。数据集格式仅支持： ``'mindrecord'`` 。可以使用 :class:`mindspore.dataset.MindDataset` 类来读取保存的 ``'mindrecord'`` 文件。

    将数据保存为 ``'mindrecord'`` 格式时存在隐式类型转换。转换表展示如何执行类型转换。

    .. list-table:: 保存为 'mindrecord' 格式时的隐式类型转换
       :widths: 25 25 50
       :header-rows: 1

       * - 'dataset'类型
         - 'mindrecord'类型
         - 说明
       * - bool
         - int32
         - 变更为int32
       * - int8
         - int32
         -
       * - uint8
         - int32
         -
       * - int16
         - int32
         -
       * - uint16
         - int32
         -
       * - int32
         - int32
         -
       * - uint32
         - int64
         -
       * - int64
         - int64
         -
       * - uint64
         - int64
         - 有可能反转
       * - float16
         - float32
         -
       * - float32
         - float32
         -
       * - float64
         - float64
         -
       * - string
         - string
         - 不支持多维字符串
       * - bytes
         - bytes
         - 不支持多维bytes

    .. note::
        1. 如需按顺序保存数据，将数据集的 `shuffle` 设置为 ``False`` ，将 `num_files` 设置为 ``1`` 。
        2. 在执行保存操作之前，不要使用batch操作、repeat操作或具有随机属性的数据增强的map操作。
        3. 当数据的维度可变时，只支持一维数组或者在第零维变化的多维数组。
        4. 不支持多维string类型、多维bytes类型。

    参数：
        - **file_name** (str) - 数据集文件的路径。
        - **num_files** (int, 可选) - 数据集文件的数量。默认值： ``1`` 。
        - **file_type** (str, 可选) - 数据集格式。默认值： ``'mindrecord'`` 。
