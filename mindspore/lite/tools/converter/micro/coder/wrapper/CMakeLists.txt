if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "-DDebug -g -fvisibility=default ${CMAKE_C_FLAGS}")
else()
    set(CMAKE_C_FLAGS "-fPIC -fPIE -Werror -O3 -fstack-protector-strong -fomit-frame-pointer ${CMAKE_C_FLAGS}")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -ffunction-sections -fdata-sections")

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    add_compile_definitions(_GNU_SOURCE)
    add_compile_definitions(ENABLE_NEON)
    add_compile_definitions(ENABLE_ARM)
endif()

if(PLATFORM_ARM64)
    add_compile_definitions(ENABLE_ARM64)
elseif(PLATFORM_ARM32)
    add_compile_definitions(ENABLE_ARM32)
else()
    set(CMSIS_DIR ${CMAKE_BINARY_DIR}/cmsis)
    message("build cmsis kernels")
    include_directories(${CMSIS_DIR}/CMSIS/Core/Include)
    include_directories(${CMSIS_DIR}/CMSIS/DSP/Include)
    include_directories(${CMSIS_DIR}/CMSIS/NN/Include)

    file(REMOVE ${CMSIS_DIR}/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c)
    file(GLOB CMSIS_OPS
            ${CMSIS_DIR}/CMSIS/NN/Source/BasicMathFunctions/*.c
            ${CMSIS_DIR}/CMSIS/NN/Source/ActivationFunctions/*.c
            ${CMSIS_DIR}/CMSIS/NN/Source/ConcatenationFunctions/*.c
            ${CMSIS_DIR}/CMSIS/NN/Source/ConvolutionFunctions/*.c
            ${CMSIS_DIR}/CMSIS/NN/Source/FullyConnectedFunctions/*.c
            ${CMSIS_DIR}/CMSIS/NN/Source/NNSupportFunctions/*.c
            ${CMSIS_DIR}/CMSIS/NN/Source/PoolingFunctions/*.c
            ${CMSIS_DIR}/CMSIS/NN/Source/ReshapeFunctions/*.c
            ${CMSIS_DIR}/CMSIS/NN/Source/SoftmaxFunctions/*.c
            )
    add_library(cmsis_nn STATIC ${CMSIS_OPS})
endif()

include(${MICRO_DIR}/cmake/package_wrapper.cmake)

# generate static library
add_library(wrapper_mid OBJECT ${WRAPPER_SRC})
add_library(wrapper STATIC $<TARGET_OBJECTS:wrapper_mid>)
target_link_libraries(wrapper_mid -lpthread)
