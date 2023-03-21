include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)

file(GLOB KERNEL_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/nnacl/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/base/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/fp32/*.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/fp32/online_fusion/*.cc
    )
if(NOT MSLITE_ENABLE_RUNTIME_PASS)
  list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp32/shape_fusion_fp32.cc)
endif()

if(PLATFORM_ARM AND MSLITE_ENABLE_FP16)
    list(REMOVE_ITEM KERNEL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/fp32/cast_for_x86_fp16.cc)
endif()

add_library(cast_for_converter_mid OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/fp32/cast_for_x86_fp16.cc)
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

if(NOT PLATFORM_ARM64)
    set(KERNEL_SRC_ARM64_FILE   ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_im2col_arm64_fp32.cc
                                ${CMAKE_CURRENT_SOURCE_DIR}/fp32/matmul_fp32_arm64.cc
                                ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_winograd_arm64_fp32.cc
    )
    list(REMOVE_ITEM KERNEL_SRC ${KERNEL_SRC_ARM64_FILE})
endif()

if(NOT PLATFORM_ARM32)
    set(KERNEL_SRC_ARM32_FILE   ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_im2col_arm32_fp32.cc
                                ${CMAKE_CURRENT_SOURCE_DIR}/fp32/matmul_fp32_arm32.cc
                                ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_winograd_arm32_fp32.cc
    )
    list(REMOVE_ITEM KERNEL_SRC ${KERNEL_SRC_ARM32_FILE})
endif()

if(NOT("${X86_64_SIMD}" STREQUAL "sse" OR "${X86_64_SIMD}" STREQUAL "avx" OR "${X86_64_SIMD}" STREQUAL "avx512"))
    set(KERNEL_SRC_SSE_FILE ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_im2col_sse_fp32.cc
                            ${CMAKE_CURRENT_SOURCE_DIR}/fp32/matmul_fp32_sse.cc
                            ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_winograd_sse_fp32.cc
    )
    list(REMOVE_ITEM KERNEL_SRC ${KERNEL_SRC_SSE_FILE})
endif()

if(NOT("${X86_64_SIMD}" STREQUAL "avx" OR "${X86_64_SIMD}" STREQUAL "avx512"))
    set(KERNEL_SRC_AVX_FILE ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_im2col_avx_fp32.cc
                            ${CMAKE_CURRENT_SOURCE_DIR}/fp32/matmul_fp32_avx.cc
                            ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_slidewindows_avx_fp32.cc
                            ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_winograd_avx_fp32.cc
    )
    list(REMOVE_ITEM KERNEL_SRC ${KERNEL_SRC_AVX_FILE})
endif()

if(NOT("${X86_64_SIMD}" STREQUAL "avx512"))
    set(KERNEL_SRC_AVX512_FILE  ${CMAKE_CURRENT_SOURCE_DIR}/fp32/convolution_im2col_avx512_fp32.cc
                                {CMAKE_CURRENT_SOURCE_DIR}/fp32/matmul_fp32_avx512.cc
    )
    list(REMOVE_ITEM KERNEL_SRC ${KERNEL_SRC_AVX512_FILE})
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

if(SUPPORT_TRAIN OR MSLITE_ENABLE_CLOUD_FUSION_INFERENCE OR MSLITE_ENABLE_CLOUD_INFERENCE)
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
