mindspore.load_obf_params_into_net
==================================

.. py:function:: mindspore.load_obf_params_into_net(network, target_modules, obf_ratios, **kwargs)

    把混淆系数导入混淆网络，通常和 :func:`mindspore.obfuscate_ckpt` 接口配合使用。

    参数：
        - **network** (nn.Cell) - 待混淆的模型网络。
        - **target_modules** (list[str, str]) - 混淆目标模块。第一个字符串表示目标模块在原来网络中的结构路径，应该写成 ``'A/B/C'`` 的形式；第二个字符串表示具体的混淆模块，应该写成 ``'A|B|C'`` 的形式。以　`GPT2 <https://gitee.com/mindspore/mindformers/blob/r0.8/mindformers/models/gpt2/gpt2.py>`_ 模型为例，它的 `target_modules` 可以是　``['backbone/blocks/attention', 'dense1|dense2|dense3']`` 。如果 `target_modules` 是 ``None`` ，该接口会自动搜索目标模块。如果搜索到了，就会对搜索到的模块进行混淆，否则会通过warning日志给出建议的混淆目标模块。
        - **obf_ratios** (Tensor) - 使用 :func:`mindspore.obfuscate_ckpt` 接口生成的混淆系数。

        - **kwargs** (dict) - 配置选项字典。

          - **ignored_func_decorators** (list[str]) - `network` 的python代码中使用到的函数装饰器名称列表。
          - **ignored_class_decorators** (list[str]) - `network` 的python代码中使用到的类装饰器名称列表。

    异常：
        - **TypeError** - `network` 不是nn.Cell类型。
        - **TypeError** - `obf_ratios` 不是Tensor类型。
        - **TypeError** - `target_modules` 不是list。
        - **TypeError** - `target_modules` 里面的元素不是string类型。
        - **ValueError** - `target_modules` 的元素个数不是 ``2`` 。
        - **ValueError** - `obf_ratios` 是空Tensor。
        - **ValueError** - `target_modules` 的第一个元素包含了除大写字母、小写字母、数字、下划线、斜杠之外的字符。
        - **ValueError** - `target_modules` 的第二个元素包含了除大写字母、小写字母、数字、下划线、竖线之外的字符。
        - **TypeError** - `ignored_func_decorators` 或者 `ignored_class_decorators` 不是string列表。
