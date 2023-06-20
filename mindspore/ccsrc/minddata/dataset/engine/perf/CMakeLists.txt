file(GLOB_RECURSE _CURRENT_SRC_FILES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cc")
set_property(SOURCE ${_CURRENT_SRC_FILES} PROPERTY COMPILE_DEFINITIONS SUBMODULE_ID=mindspore::SubModuleId::SM_MD)
add_library(engine-perf OBJECT
        auto_tune.cc
        connector_size.cc
        cpu_sampler.cc
        dataset_iterator_tracing.cc
        device_queue_tracing.cc
        info_collector.cc
        monitor.cc
        profiling.cc
        )
