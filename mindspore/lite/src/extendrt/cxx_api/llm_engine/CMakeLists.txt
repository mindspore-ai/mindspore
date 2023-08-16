include_directories(${TOP_DIR}/graphengine/910/metadef/inc/external)
include_directories(${TOP_DIR}/graphengine/910/metadef/inc)
include_directories(${TOP_DIR}/graphengine/910/inc)
include_directories(${TOP_DIR}/graphengine/910/inc/external)
include_directories(${TOP_DIR}/graphengine/910/third_party/fwkacllib)
include_directories(${TOP_DIR}/graphengine/910/third_party/fwkacllib/inc)
include_directories(${CCSRC_DIR})

file(STRINGS "${TOP_DIR}/version.txt" MSVERSION)
add_definitions(-DMSVERSION=\"${MSVERSION}\")
add_compile_definitions(ENABLE_SECURITY)

file(GLOB LLM_ENGINE_PLUGIN_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/llm_engine_plugin.cc
    ${CCSRC_DIR}/utils/config_manager.cc
    )
list(APPEND LLM_ENGINE_PLUGIN_SRC $<TARGET_OBJECTS:_mindspore_transform_graph_ir_obj>)

add_library(llm_engine_plugin SHARED ${LLM_ENGINE_PLUGIN_SRC})

file(GLOB LLM_ENGINE_MOCK_SRC
    ${CMAKE_CURRENT_SOURCE_DIR}/llm_engine_mock.cc
    )
add_library(llm_engine SHARED ${LLM_ENGINE_MOCK_SRC})

#find_library(llm_engine libllm_engine.so ${ASCEND_CANN_RUNTIME_PATH} ${ASCEND_TOOLKIT_RUNTIME_PATH})
target_link_libraries(llm_engine_plugin llm_engine mindspore_converter mindspore_core)

target_link_libraries(llm_engine_plugin mindspore-extendrt)
