include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)

# This set up makes the source code more portable.
include_directories(${PYTHON_INCLUDE_DIRS})

# source directory
aux_source_directory(io DIR_LIB_SRCS)
aux_source_directory(meta DIR_LIB_SRCS)
aux_source_directory(common DIR_LIB_SRCS)

# set(CMAKE_CXX_COMPILER "g++")
# set(CMAKE_CXX_FLAGS "-Wall -fvisibility=hidden")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")

# add shared link library
add_library(_c_mindrecord SHARED ${DIR_LIB_SRCS})

set_target_properties(_c_mindrecord PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

# add link library
target_link_libraries(_c_mindrecord PRIVATE mindspore::sqlite ${PYTHON_LIB} ${SECUREC_LIBRARY} mindspore mindspore_gvar mindspore::protobuf)

if (USE_GLOG)
    target_link_libraries(_c_mindrecord PRIVATE mindspore::glog)
endif()
