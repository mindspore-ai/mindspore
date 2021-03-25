cmake_minimum_required (VERSION 3.6)

project(CMSISNNFullyConnected)

include(configLib)

file(GLOB SRC "./*_*.c")

add_library(CMSISNNFullyConnected STATIC ${SRC})

configLib(CMSISNNFullyConnected ${ROOT})
configDsp(CMSISNNFullyConnected ${ROOT})

### Includes
target_include_directories(CMSISNNFullyConnected PUBLIC "${NN}/Include")
target_include_directories(CMSISNNFullyConnected PUBLIC "${ROOT}/CMSIS/DSP/Include")



