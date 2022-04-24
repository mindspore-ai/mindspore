include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/fp32/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/int8/*.cc
    )
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
list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/int8/opt_op_handler.cc)

if(SUPPORT_TRAIN)
    file(GLOB TRAIN_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp16_grad/*.cc)
    file(GLOB TRAIN_KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp32_grad/*.cc)
    add_library(train_cpu_kernel_mid OBJECT ${TRAIN_KERNEL_SRC})
    add_dependencies(train_cpu_kernel_mid fbs_src)
    if("${CMAKE_CXX_FLAGS}" MATCHES "-ffast-math")
        # using -fno-finite-math-only to avoid unexpected errors may occur due to compile optimization
        target_compile_options(train_cpu_kernel_mid PRIVATE -fno-finite-math-only)
    endif()
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
