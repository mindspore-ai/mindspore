add_library(engine-datasetops-source-sampler OBJECT
    distributed_sampler.cc
    pk_sampler.cc
    python_sampler.cc
    random_sampler.cc
    sampler.cc
    sequential_sampler.cc
    subset_random_sampler.cc
    weighted_random_sampler.cc
    )
