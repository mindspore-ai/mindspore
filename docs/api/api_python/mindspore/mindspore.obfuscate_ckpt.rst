mindspore.obfuscate_ckpt
========================

.. py:function:: mindspore.obfuscate_ckpt(network, ckpt_files, target_modules=None, obf_config=None, saved_path='./', obfuscate_scale=100)

    根据用户配置的混淆策略，对模型的Checkpoint进行混淆保护，防止攻击者窃取模型资产。

    参数：
        - **network** (nn.Cell) - 待混淆的原始网络。
        - **ckpt_files** (str) - 待混淆的原始权重文件的存储路径。
        - **target_modules** (list[str]，可选) - 需要混淆的目标算子。第一个字符串表示目标算子在原网络中的路径，需为 ``"A/B/C"`` 的形式，第二个字符串表示同一个路径下的多个目标算子名，需是 ``"D|E|F"`` 的形式。例如，GPT2的 `target_modules` 可以是 ``['backbone/blocks/attention', 'dense1|dense2|dense3']`` 。如果 `target_modules` 有第三个值，格式需是 ``"obfuscate_layers:all"`` 或 ``"obfuscate_layers:int"`` ，表示需要混淆重复层（如transformer层或resnet块）的层数。默认值： ``None`` 。
        - **obf_config** (dict，可选) - 模型混淆策略的配置。默认值： ``None`` 。
        - **saved_path** (str，可选) - 混淆后权重文件的保存路径。默认值： ``'./'`` 。
        - **obfuscate_scale** (Union[float, int]，可选) - 权重混淆尺度，控制混淆系数的取值范围为大于1的int或float。默认值： ``100``。

    返回：
        解混淆元数据（dict）。该数据在运行混淆网络时加载使用。

    异常：
        - **TypeError** - `network` 不是nn.Cell类型。
        - **TypeError** - `ckpt_files` 不是str类型或者 `saved_path` 不是str类型。
        - **TypeError** - `target_modules` 不是list类型。
        - **TypeError** - `target_modules` 中的元素不是str类型。
        - **TypeError** - `obf_config` 不是dict类型。
        - **ValueError** - `ckpt_files` 目录不存在或者 `saved_path` 目录不存在。
        - **ValueError** - `target_modules` 中的元素个数小于2。
        - **ValueError** - `target_modules` 的第一个字符串包含大小写字母、数字、 ``'_'`` 或 ``'/'`` 以外的字符。
        - **ValueError** - `target_modules` 的第二个字符串为空，或包含大小写字母、数字、 ``'_'`` 或 ``'/'`` 以外的字符。
        - **ValueError** - `target_modules` 的第三个字符串不是 ``"obfuscate_layers:all"`` 或 ``"obfuscate_layers:int"`` 的格式。
