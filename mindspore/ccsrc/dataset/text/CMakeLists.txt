add_subdirectory(kernels)

add_library(text OBJECT
        vocab.cc
        )

add_dependencies(text text-kernels)