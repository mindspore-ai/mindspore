include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../include/)
set(OP_SRC
    common_func.c.o
    common_func_int8.c.o
    conv3x3_int8.c.o
    conv_int8.c.o
    fixed_point.c.o
    matmul_int8.c.o
    matmul_int8_wrapper.c.o
    pack_int8.c.o
    pooling_int8.c.o
    quant_dtype_cast_int8.c.o
    reshape_int8.c.o
    softmax_int8.c.o
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
