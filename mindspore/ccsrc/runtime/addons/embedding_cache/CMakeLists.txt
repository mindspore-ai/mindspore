file(GLOB_RECURSE EMBEDDING_CACHE_SRC_LIST RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    )

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

set_property(SOURCE ${EMBEDDING_CACHE_SRC_LIST} PROPERTY SUBMODULE_ID=mindspore::SubModuleId::SM_RUNTIME_FRAMEWORK)
add_library(_mindspore_runtime_embedding_cache_obj OBJECT ${EMBEDDING_CACHE_SRC_LIST})
