if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(CMAKE_C_FLAGS "-DDebug -g -fvisibility=default ${CMAKE_C_FLAGS}")
else()
    set(CMAKE_C_FLAGS "-fPIC -fPIE -Werror -O3 -fstack-protector-strong -fomit-frame-pointer ${CMAKE_C_FLAGS}")
    set(CMAKE_C_FLAGS_Release "${CMAKE_C_FLAGS_Release} -O3 -ffunction-sections -fdata-sections")
    string(REPLACE "-g" "" CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
endif()

if(PLATFORM_ARM64 OR PLATFORM_ARM32)
    add_compile_definitions(ENABLE_NEON)
    add_compile_definitions(ENABLE_ARM)
endif()

if(PLATFORM_ARM64)
    add_compile_definitions(ENABLE_ARM64)
elseif(PLATFORM_ARM32)
    add_compile_definitions(ENABLE_ARM32)
else()
    include(${MICRO_DIR}/cmake/package_cmsis.cmake)
    add_library(cmsis_nn STATIC ${CMSIS_OPS})
endif()

include(${MICRO_DIR}/cmake/package_wrapper.cmake)

# generate static library
add_library(wrapper STATIC ${WRAPPER_SRC})
