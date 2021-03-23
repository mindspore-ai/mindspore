cmake_minimum_required (VERSION 3.6)

project(CMSISNNSupport)

include(configLib)

file(GLOB SRC "./*_*.c")

add_library(CMSISNNSupport STATIC ${SRC})

configLib(CMSISNNSupport ${ROOT})
configDsp(CMSISNNSupport ${ROOT})

### Includes
target_include_directories(CMSISNNSupport PUBLIC "${NN}/Include")
target_include_directories(CMSISNNSupport PUBLIC "${ROOT}/CMSIS/DSP/Include")



