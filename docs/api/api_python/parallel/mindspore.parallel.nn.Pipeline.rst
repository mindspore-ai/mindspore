mindspore.parallel.nn.Pipeline
============================================================================

.. py:class:: mindspore.parallel.nn.Pipeline(network, micro_size, stage_config=None, segment_config=None)

    指定流水线并行的micro_batch个数、stage和segment的划分规则。

    .. note::
        `micro_size` 必须大于等于 `pipeline_stages`。

    参数：
        - **network** (Cell) - 将进行pp并行的网络。
        - **micro_size** (int) - MicroBatchsize。
        - **stage_config** (dict，可选) - 流水线并行对于每个cell的stage配置。默认值： ``None``。
        - **segment_config** (dict，可选) - 流水线并行对于每个cell的segment配置。默认值： ``None``。

    异常：
        - **TypeError** - `network` 不是cell类型输入。
        - **TypeError** - `micro_size` 不是整数类型。
        - **ValueError** - `micro_size` 值异常，为0或者负数。
        - **KeyError** - `dict` cell名称匹配异常，遍历当前 `network` 下所有 `cell` 仍有剩余的配置项。
