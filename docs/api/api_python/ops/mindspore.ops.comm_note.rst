运行以下样例之前，需要配置好通信环境变量。

针对Ascend设备，用户需要准备rank表，设置rank_id和device_id，详见 `rank table启动 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/rank_table.html>`_ 。

针对GPU设备，用户需要准备host文件和mpi，详见 `mpirun启动 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/mpirun.html>`_ 。

针对CPU设备，用户需要编写动态组网启动脚本，详见 `动态组网启动 <https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/dynamic_cluster.html>`_ 。