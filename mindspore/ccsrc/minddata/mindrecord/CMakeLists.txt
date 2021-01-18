include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)

# This set up makes the source code more portable.
include_directories(${PYTHON_INCLUDE_DIRS})

# source directory
aux_source_directory(io DIR_LIB_SRCS)
aux_source_directory(meta DIR_LIB_SRCS)
aux_source_directory(common DIR_LIB_SRCS)

# set(CMAKE_CXX_COMPILER "g++")
# set(CMAKE_CXX_FLAGS "-Wall -fvisibility=hidden")
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-command-line-argument")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-return-type")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")

# add shared link library
set_property(SOURCE ${DIR_LIB_SRCS} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(_c_mindrecord SHARED ${DIR_LIB_SRCS})

set_target_properties(_c_mindrecord PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

# add link library
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    target_link_libraries(_c_mindrecord PRIVATE mindspore::sqlite mindspore mindspore_gvar mindspore::protobuf)
else()
    target_link_libraries(_c_mindrecord PRIVATE mindspore::sqlite ${PYTHON_LIB} ${SECUREC_LIBRARY} mindspore
                                                mindspore_gvar mindspore::protobuf)
endif()

if(USE_GLOG)
    target_link_libraries(_c_mindrecord PRIVATE mindspore::glog)
else()
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_options(_c_mindrecord PRIVATE -Wl,-init,mindspore_log_init)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set_target_properties(_c_mindrecord PROPERTIES MACOSX_RPATH ON)
    endif()
endif()

set_target_properties(_c_mindrecord PROPERTIES INSTALL_RPATH ${MINDSPORE_RPATH})
