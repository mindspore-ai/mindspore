
.. py:method:: get_num_samples()

    获取当前采样器实例的num_samples参数值。此参数在定义Sampler时，可以选择性传入（默认为None）。此方法将返回num_samples的值。如果当前采样器有子采样器，会继续访问子采样器，并根据一定的规则处理获取值。

    下表显示了各种可能的组合，以及最终返回的结果。

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

    返回：
        int，样本数，可为None。