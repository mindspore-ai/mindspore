mindspore.parallel.restore_group_info_list
============================================================================

.. py:function:: mindspore.parallel.restore_group_info_list(group_info_file_name)

    从通信域文件中提取rank list信息。通过配置环境变量 `GROUP_INFO_FILE` 以在编译阶段存储该通信域信息，例如 "export GROUP_INFO_FILE=/data/group_info.pb"。

    参数：
        - **group_info_file_name** (str) - 保存通信域的文件名称。

    返回：
        List，通信域列表。

    异常：
        - **ValueError** - 通信域文件格式不正确。
        - **TypeError** - `group_info_file_name` 不是字符串。