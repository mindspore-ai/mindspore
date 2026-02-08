mindspore.parallel.parameter_broadcast
============================================================================

.. py:function:: mindspore.parallel.parameter_broadcast(net, layout, cur_rank=0, initial_rank=0)

    在数据并行维度，将参数广播到其他卡上。

    .. warning::
        实验性接口，未来可能变更或移除。

    参数：
        - **net** (Cell) - 将进行参数广播的网络。
        - **layout** (Dict) - 参数layout字典，来自于函数 :func:`mindspore.nn.Cell.parameter_layout_dict` ；也可以从从文件中读取。例如，通过 :func:`mindspore.parallel.auto_parallel.AutoParallel.save_param_strategy_file` 保存下来的strategy.ckpt。该字典的key是参数名称，value是参数的Layout。
        - **cur_rank** (int，可选) - 当前的rankID。默认值： ``0`` 。
        - **initial_rank** (int，可选) - 每个流水线并行阶段的起始rankID。默认值： ``0`` 。

    异常：
        - **ValueError** - `cur_rank` 不是当前rank的rankID。
        - **ValueError** - `initial_rank` 不是当前流水线并行阶段的起始rankID。
        - **ValueError** - `layout` 中的参数名称不在函数 :func:`mindspore.nn.Cell.parameters_dict` 。