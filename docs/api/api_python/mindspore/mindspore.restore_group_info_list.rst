mindspore.restore_group_info_list
=======================================

.. py:function:: mindspore.restore_group_info_list(group_info_file_name)

    从group_info_file_name指向的文件中提取得到通信域的信息，在该通信域内的所有设备的checkpoint文件均与存储group_info_file_name的设备相同，可以直接进行替换。通过配置环境变量GROUP_INFO_FILE以在编译阶段存储下该通信域信息，例如"export GROUP_INFO_FILE=/data/group_info.pb"。

    参数：
        - **group_info_file_name** (str) - 保存通信域的文件的名字。

    返回：
        List，通信域列表。

    异常：
        - **ValueError** - 通信域文件格式不正确。
        - **TypeError** - `group_info_file_name` 不是字符串。
