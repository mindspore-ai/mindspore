include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/include)
include_directories(${CMAKE_BINARY_DIR})

if(ENABLE_DEBUGGER)
    # debugger: compile proto files
    include_directories("${CMAKE_BINARY_DIR}/debug/debugger")
endif()

# ===== CPU
add_subdirectory(cpu)

# ===== Host
add_subdirectory(host)

# ===== Ascend
if(ENABLE_D)
    add_subdirectory(ascend)
endif()

# ===== GPU
if(ENABLE_GPU)
    add_subdirectory(gpu)
    enable_directory_when_only_build_plugins(gpu)
endif()