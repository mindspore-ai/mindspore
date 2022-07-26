mindspore_lite.RunnerConfig
===========================

.. py:class:: mindspore_lite.RunnerConfig(context=None, workers_num=None, config_info=None)

    RunnerConfig类定义一个或多个Servables的runner config。
    该类可用于模型的并行推理，与模型提供的服务相对应。
    客户端通过往服务器发送推理任务并接收推理结果。

    参数：
        - **context** (Context，可选) - 定义用于在执行期间存储选项的上下文。默认值：None。
        - **workers_num** (int，可选) - workers的数量。默认值：None。
        - **config_info** (dict{str, dict{str, str}}，可选) - 传递模型权重文件路径的嵌套映射。例如：{"weight": {"weight_path": "/home/user/weight.cfg"}}。默认值：None。key当前支持["weight"]；value为dict格式，其中的key当前支持["weight_path"]，其中的value为权重的路径，例如"/home/user/weight.cfg"。

    异常：
        - **TypeError** - `context` 不是Context类型或None。
        - **TypeError** - `workers_num` 不是int类型或None。
        - **TypeError** - `config_info` 不是dict类型或None。
        - **TypeError** - `config_info` 是dict类型，但key不是str类型。
        - **TypeError** - `config_info` 是dict类型，key是str类型，但value不是dict类型。
        - **TypeError** - `config_info` 是dict类型，key是str类型，value是dict类型，但value的key不是str类型。
        - **TypeError** - `config_info` 是dict类型，key是str类型，value是dict类型，value的key是str类型，但value的value不是str类型。
        - **ValueError** - `workers_num` 是int类型，但小于0。
