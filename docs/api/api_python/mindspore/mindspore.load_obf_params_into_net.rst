mindspore.load_obf_params_into_net
==================================

.. py:function:: mindspore.load_obf_params_into_net(network, target_modules=None, obf_ratios=None, obf_config=None, data_parallel_num=1, **kwargs)

    根据用户配置的混淆策略，对模型结构进行修改，并将混淆态Checkpoint加载到模型中。

    参数：
        - **network** (nn.Cell) - 待混淆的原始网络。
        - **target_modules** (list[str]，可选) - 需要混淆的目标算子。第一个字符串表示目标算子在原网络中的路径，需为 ``"A/B/C"`` 的形式。第二个字符串表示同一个路径下的多个目标算子名，需为 ``"D|E|F"`` 的形式。例如，GPT2的 `target_modules` 可以是 ``['backbone/blocks/attention', 'dense1|dense2|dense3']`` 。如果 `target_modules` 有第三个值，它的格式需为 ``"obfuscate_layers:all"`` 或 ``"obfuscate_layers:int"`` ，这表示需要混淆重复层（如transformer层或resnet块）的层数。默认值： ``None`` 。
        - **obf_ratios** (Tensor，可选) - 混淆系数，由 `mindspore.obfuscate_ckpt` 接口生成。默认值： ``None`` 。
        - **obf_config** (dict，可选) - 模型混淆策略的配置。默认值： ``None`` 。
        - **data_parallel_num** (int，可选) - 模型并行训练的数据并行度。默认值： ``1`` 。
        - **kwargs** (dict) - 配置选项字典。

          - **ignored_func_decorators** (list[str]) - Python代码中函数装饰器的名字列表。
          - **ignored_class_decorators** (list[str]) - Python代码中类装饰器的名字列表。

    返回：
        混淆后模型(nn.Cell)。

    异常：
        - **TypeError** - `network` 不是nn.Cell类型。
        - **TypeError** - `obf_ratios` 不是Tensor类型。
        - **TypeError** - `target_modules` 不是list类型。
        - **TypeError** - `obf_config` 不是dict类型。
        - **TypeError** - `target_modules` 中的元素不是str类型。
        - **ValueError** - `obf_ratios` 为空。
        - **ValueError** - `target_modules` 中的元素个数小于2。
        - **ValueError** - `target_modules` 的第一个字符串包含大小写字母、数字、 ``'_'`` 和 ``'/'`` 以外的字符。
        - **ValueError** - `target_modules` 的第二个字符串为空或包含大小写字母、数字、 ``'_'`` 和 ``'/'`` 以外的字符。
        - **ValueError** - `target_modules` 的第三个字符串不是 ``"obfuscate_layers:all"`` 或 ``"obfuscate_layers:int"`` 的格式。
        - **TypeError** - `ignored_func_decorators` 不是字符串列表，或 `ignored_class_decorators` 不是字符串列表。
    