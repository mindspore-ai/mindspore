
.. py:method:: get_num_samples()

    返回采样器采集样本数量，如果存在子采样器，则子采样器计数可以是数值或None。这些条件会影响最终的采样结果。

    下表显示了调用此函数的可能结果。

    .. list-table::
        :widths: 25 25 25 25
        :header-rows: 1

        * - 子采样器
          - num_samples
          - child_samples
          - 结果
        * - T
          - x
          - y
          - min(x, y)
        * - T
          - x
          - None
          - x
        * - T
          - None
          - y
          - y
        * - T
          - None
          - None
          - None
        * - None
          - x
          - n/a
          - x
        * - None
          - None
          - n/a
          - None

    **返回：**

    int，样本数，可为None。