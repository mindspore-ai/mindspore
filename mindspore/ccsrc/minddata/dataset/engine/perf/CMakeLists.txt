add_library(engine-perf OBJECT
    profiling.cc
    monitor.cc
    device_queue_tracing.cc
    connector_size.cc
    dataset_iterator_tracing.cc
    connector_throughput.cc
    cpu_sampling.cc
        )
