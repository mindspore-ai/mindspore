mindspore.load_param_into_net
=============================

.. py:function:: mindspore.load_param_into_net(net, parameter_dict, strict_load=False)

    将参数加载到网络中，返回网络中没有被加载的参数列表。

    参数：
        - **net** (Cell) - 将要加载参数的网络。
        - **parameter_dict** (dict) - 加载checkpoint文件得到的字典。
        - **strict_load** (bool) - 是否将参数严格加载到网络中。如果是False, 它将以相同的后缀名将参数字典中的参数加载到网络中，并会在精度不匹配时，进行精度转换，比如将 `float32` 转换为 `float16` 。默认值：False。

    返回：
        - **param_not_load** (List)，网络中没有被加载的参数。
        - **ckpt_not_load** (List)，checkpoint文件中没有被加载的参数。

    异常：
        - **TypeError** - 如果参数不是Cell，或者 `parameter_dict` 不是Parameter类型的字典。
