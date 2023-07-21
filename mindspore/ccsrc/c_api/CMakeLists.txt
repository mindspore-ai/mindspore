file(GLOB_RECURSE C_API_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "./*.cc"
        )

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(mindspore_c_api_obj OBJECT ${C_API_SRC_LIST})
add_library(mindspore_c_api SHARED $<TARGET_OBJECTS:mindspore_c_api_obj>)

if(MS_BUILD_GRPC)
    target_link_libraries(mindspore_c_api PRIVATE mindspore::grpc++)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(mindspore_c_api PRIVATE -Wl,--no-as-needed ${PYTHON_LIBRARIES})
else()
    target_link_libraries(mindspore_c_api PRIVATE ${PYTHON_LIBRARIES})
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    add_compile_definitions(BUILDING_C_API_DLL)
endif()

target_link_libraries(mindspore_c_api PRIVATE ${SECUREC_LIBRARY} mindspore_backend mindspore mindspore_common
        proto_input mindspore::protobuf)
# On ARM platform, the call of infer API will be treated as the cross-library call from frontend by compiler. Thus, the
# link mode should be be PUBLIC or else a segmentation fault will occur.
target_link_libraries(mindspore_c_api PUBLIC mindspore_core)

if(ENABLE_CPU)
    target_link_libraries(mindspore_c_api PRIVATE mindspore::dnnl mindspore::mkldnn)
endif()

if(USE_GLOG)
    target_link_libraries(mindspore_c_api PRIVATE mindspore::glog)
endif()
