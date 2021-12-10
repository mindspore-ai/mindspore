file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(
        engine-perf OBJECT
        profiling.cc
        monitor.cc
        device_queue_tracing.cc
        connector_size.cc
        dataset_iterator_tracing.cc
        cpu_sampler.cc
        auto_tune.cc
)
