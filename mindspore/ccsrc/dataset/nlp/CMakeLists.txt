add_subdirectory(kernels)

add_library(nlp OBJECT
        vocab.cc
        )

add_dependencies(nlp nlp-kernels)