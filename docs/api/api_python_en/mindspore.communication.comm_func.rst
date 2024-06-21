mindspore.communication.comm_func
=================================
Collection communication functional interface

Note that the APIs in the following list need to preset communication environment variables.

For Ascend/GPU/CPU devices, it is recommended to use the msrun startup method
without any third-party or configuration file dependencies.
Please see the `msrun start up
<https://www.mindspore.cn/tutorials/experts/zh-CN/master/parallel/msrun_launcher.html>`_
for more details.

.. msplatformautosummary::
    :toctree: communication
    :nosignatures:
    :template: classtemplate.rst

    mindspore.communication.comm_func.all_gather_into_tensor
    mindspore.communication.comm_func.all_reduce
    mindspore.communication.comm_func.all_to_all_single_with_output_shape
    mindspore.communication.comm_func.all_to_all_with_output_shape
    mindspore.communication.comm_func.barrier
    mindspore.communication.comm_func.batch_isend_irecv
    mindspore.communication.comm_func.broadcast
    mindspore.communication.comm_func.gather_into_tensor
    mindspore.communication.comm_func.irecv
    mindspore.communication.comm_func.isend
    mindspore.communication.comm_func.P2POp
    mindspore.communication.comm_func.reduce
    mindspore.communication.comm_func.reduce_scatter_tensor
    mindspore.communication.comm_func.scatter_tensor
