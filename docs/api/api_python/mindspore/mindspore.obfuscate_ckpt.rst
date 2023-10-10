mindspore.obfuscate_ckpt
========================

.. py:function:: mindspore.obfuscate_ckpt(network, ckpt_files, target_modules=None, saved_path='./')

    对明文的ckpt权重文件进行混淆，通常和 :func:`mindspore.load_obf_params_into_net` 接口配合使用。

    参数：
        - **network** (nn.Cell) - 待混淆的模型网络。
        - **ckpt_files** (str) - 待混淆的模型ckpt文件存放目录。
        - **target_modules** (list[str, str]) - 混淆目标模块。第一个字符串表示目标模块在原来网络中的结构路径，应该写成 ``'A/B/C'`` 的形式；第二个字符串表示具体的混淆模块，应该写成 ``'A|B|C'`` 的形式。以　`GPT2 <https://gitee.com/mindspore/mindformers/blob/r0.8/mindformers/models/gpt2/gpt2.py>`_ 模型为例，它的 `target_modules` 可以是　``['backbone/blocks/attention', 'dense1|dense2|dense3']`` 。如果 `target_modules` 是 ``None`` ，该接口会自动搜索目标模块。如果搜索到了，就会对搜索到的模块进行混淆，否则会通过warning日志给出建议的混淆目标模块。默认值： ``None`` 。
        - **saved_path** (str) - 混淆后的ckpt文件和混淆系数（numpy文件格式）的保存路径，混淆系数是在执行混淆网络时必须导入的数据。默认值： ``./`` 。

    异常：
        - **TypeError** - `network` 不是nn.Cell类型。
        - **TypeError** - `ckpt_files` 不是string类型或者 `saved_path` 不是string类型。
        - **TypeError** - `target_modules` 不是list类型。
        - **TypeError** - `target_modules` 里面的元素不是string类型。
        - **ValueError** - `ckpt_files` 路径不存在或者 `saved_path`　路径不存在。
        - **ValueError** - `target_modules` 的元素个数不是 ``2`` 。
        - **ValueError** - `target_modules` 的第一个元素包含了除大写字母、小写字母、数字、下划线、斜杠之外的字符。
        - **ValueError** - `target_modules` 的第二个元素包含了除大写字母、小写字母、数字、下划线、竖线之外的字符。

