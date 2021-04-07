include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/)
include_directories(${OP_HEADER_PATH}/CMSIS/NN/Include)
include_directories(${OP_HEADER_PATH}/CMSIS/DSP/Include)
include_directories(${OP_HEADER_PATH}/CMSIS/Core/Include)
set(OP_SRC
    arm_convolve_s8.c.o
    arm_fully_connected_s8.c.o
    arm_max_pool_s8.c.o
    arm_nn_mat_mult_kernel_s8_s16.c.o
    arm_nn_vec_mat_mult_t_s8.c.o
    arm_q7_to_q15_with_offset.c.o
    arm_softmax_s8.c.o
    quant_dtype_cast_int8.c.o
    weight.c.o
    net.c.o
    session.cc.o
    tensor.cc.o
    string.cc.o
)
file(GLOB NET_SRC
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
     ${CMAKE_CURRENT_SOURCE_DIR}/*.c
     )
add_library(net STATIC ${NET_SRC})
