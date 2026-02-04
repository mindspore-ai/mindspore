mindspore.get_ckpt_path_with_strategy
======================================

.. py:function:: mindspore.get_ckpt_path_with_strategy(cur_ckpt_path, cur_strategy_path)

    从当前卡的所有存在备份关系的checkpoint文件中找到可用的checkpoint文件。

    该接口假定文件名中存在rank_{rank_id}的字符串用于区分不同的checkpoint文件。如果不存在该字符串并且传入路径可用，则返回传入路径，否则返回None。

    .. note::
        - 这个API必须在集群初始完成后调用，因为接口内部需要获取集群信息。

    参数：
        - **cur_ckpt_path** (str) - 当前卡需要的checkpoint的文件路径。
        - **cur_strategy_path** (str) - 当前卡的strategy文件。

    返回：
        - str，如果找到可用checkpoint文件，返回该路径。
        - None，如果未找到可用checkpoint文件，返回None。
