file(GLOB_RECURSE C_API_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "./src/helper.cc"
        "./src/tensor.cc"
        "./src/node.cc"
        "./src/graph.cc"
        "./src/attribute.cc"
        "./src/abstract.cc"
        "./src/resource_manager.cc"
        "./src/context.cc"
        "./src/attribute.cc"
        "./src/utils.cc"
        )

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

if(ENABLE_D OR ENABLE_GPU)
    target_link_libraries(mindspore_c_api PRIVATE ${SECUREC_LIBRARY} mindspore_backend mindspore_core
            mindspore_common proto_input mindspore::protobuf)
else()
    target_link_libraries(mindspore_c_api PRIVATE ${SECUREC_LIBRARY} mindspore_backend mindspore
            mindspore_core mindspore_common proto_input mindspore::protobuf)
endif()

if(ENABLE_CPU)
    target_link_libraries(mindspore_c_api PRIVATE mindspore::dnnl mindspore::mkldnn)
endif()

if(USE_GLOG)
    target_link_libraries(mindspore_c_api PRIVATE mindspore::glog)
endif()
