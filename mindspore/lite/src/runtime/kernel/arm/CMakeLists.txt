include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/fp32/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/int8/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/string/*.cc
    )
list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/int8/opt_op_handler.cc)

if(SUPPORT_TRAIN)
    file(GLOB TRAIN_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp16_grad/*.cc)
    file(GLOB TRAIN_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp32_grad/*.cc)
    set(KERNEL_SRC ${KERNEL_SRC} ${TRAIN_KERNEL_SRC})
endif()

add_library(cpu_kernel_mid OBJECT ${KERNEL_SRC})
add_dependencies(cpu_kernel_mid fbs_src)

if(PLATFORM_ARM64)
    if(ENABLE_FP16)
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
