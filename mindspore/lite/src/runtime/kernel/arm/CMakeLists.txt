include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/fp32/*.cc
    )
if(NOT MSLITE_ENABLE_RUNTIME_PASS)
  list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp32/shape_fusion_fp32.cc)
endif()

if(PLATFORM_ARM AND MSLITE_ENABLE_FP16)
    list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp32/cast_for_converter_fp32.cc)
endif()

add_library(cast_for_converter_mid OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/fp32/cast_for_converter_fp32.cc)
add_dependencies(cast_for_converter_mid fbs_src)

if(MSLITE_ENABLE_INT8)
    file(GLOB INT8_KERNEL_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/int8/*.cc
            )
    set(KERNEL_SRC
            ${KERNEL_SRC}
            ${INT8_KERNEL_SRC}
            )
    list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/int8/opt_op_handler.cc)
else()
    list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/base/quant_dtype_cast.cc)
endif()

if(MSLITE_ENABLE_SPARSE_COMPUTE)
    file(GLOB SPARSE_KERNEL_SRC
            ${CMAKE_CURRENT_SOURCE_DIR}/fp32_sparse/*.cc
            )
    set(KERNEL_SRC
            ${KERNEL_SRC}
            ${SPARSE_KERNEL_SRC}
            )
endif()

if(MSLITE_ENABLE_STRING_KERNEL)
    file(GLOB KERNEL_STRING_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/string/*.cc
        )
    set(KERNEL_SRC
        ${KERNEL_SRC}
        ${KERNEL_STRING_SRC}
        )
endif()
if(MSLITE_ENABLE_CONTROLFLOW)
    file(GLOB KERNEL_CONTROL_TENSORLIST
            ${CMAKE_CURRENT_SOURCE_DIR}/control/*.cc
            )
    set(KERNEL_SRC
            ${KERNEL_SRC}
            ${KERNEL_CONTROL_TENSORLIST}
            )
endif()

if(SUPPORT_TRAIN)
    file(GLOB TRAIN_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp16_grad/*.cc)
    file(GLOB TRAIN_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp32_grad/*.cc)
    set(KERNEL_SRC ${KERNEL_SRC} ${TRAIN_KERNEL_SRC})
endif()

add_library(cpu_kernel_mid OBJECT ${KERNEL_SRC})
add_dependencies(cpu_kernel_mid fbs_src)
if(PLATFORM_ARM)
    if(MSLITE_ENABLE_FP16)
        file(GLOB FP16_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp16/*.cc)
        if(SUPPORT_TRAIN)
            file(GLOB FP16_KERNEL_TRAIN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp16_grad/*.cc)
            set(FP16_KERNEL_SRC ${FP16_KERNEL_SRC} ${FP16_KERNEL_TRAIN_SRC})
        endif()
        add_library(cpu_fp16_kernel_mid OBJECT ${FP16_KERNEL_SRC})
        add_dependencies(cpu_fp16_kernel_mid fbs_src)
    endif()
    file(GLOB OPT_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/int8/opt_op_handler.cc)
    add_library(cpu_opt_kernel_mid OBJECT ${OPT_KERNEL_SRC})
    add_dependencies(cpu_kernel_mid fbs_src)
endif()
