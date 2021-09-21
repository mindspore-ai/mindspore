add_library(
        engine-perf OBJECT
        profiling.cc
        monitor.cc
        device_queue_tracing.cc
        connector_size.cc
        dataset_iterator_tracing.cc
        cpu_sampler.cc
)
