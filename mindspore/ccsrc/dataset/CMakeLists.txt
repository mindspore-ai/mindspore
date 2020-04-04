set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sequence-point")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

############################# Options ################################
if (ENABLE_GPUQUE)
    add_definitions(-D ENABLE_GPUQUE)
    message(STATUS "GPU queue is enabled")
endif ()
if (ENABLE_TDTQUE)
    add_definitions(-D ENABLE_TDTQUE)
    message(STATUS "TDT queue is enabled")
endif ()

# conde coverage
# option(ENABLE_COVERAGE "Enable code coverage report" OFF)
# if (ENABLE_COVERAGE)
#   include(${CMAKE_SOURCE_DIR}/cmake/CodeCoverage.cmake)
#   append_coverage_compiler_flags()
# endif ()

########### Set up the include directories ###########################
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/device/ascend/platform)

include_directories(${CMAKE_BINARY_DIR}) # for protobuf generated .h

include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/mindrecord/include)
######################################################################

####################### Flags ########################################
# compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")

################## Include sub-modules ###############################
add_subdirectory(util)
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(engine)
add_subdirectory(api)
######################################################################

################### Create _c_dataengine Library ######################
set(submodules
    $<TARGET_OBJECTS:core>
    $<TARGET_OBJECTS:utils>
    $<TARGET_OBJECTS:kernels>
    $<TARGET_OBJECTS:kernels-image>
    $<TARGET_OBJECTS:kernels-data>
    $<TARGET_OBJECTS:APItoPython>
    $<TARGET_OBJECTS:engine-datasetops-source>
    $<TARGET_OBJECTS:engine-datasetops-source-sampler>
    $<TARGET_OBJECTS:engine-datasetops>
    $<TARGET_OBJECTS:engine>
    )

if (ENABLE_TDTQUE)
    add_library(_c_dataengine SHARED ${submodules} $<TARGET_OBJECTS:engine-tdt>)
else ()
    add_library(_c_dataengine SHARED ${submodules})
endif ()


set_target_properties(_c_dataengine PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )

######################################################################

################# Link with external libraries ########################
target_link_libraries(_c_dataengine PRIVATE mindspore mindspore_gvar)
target_link_libraries(_c_dataengine PRIVATE mindspore::pybind11_module -ldl protobuf::libprotobuf ${SECUREC_LIBRARY})
target_link_libraries(_c_dataengine PUBLIC mindspore::jpeg_turbo mindspore::opencv_core mindspore::opencv_imgcodecs
        mindspore::opencv_imgproc)
if (ENABLE_GPUQUE)
    target_link_libraries(_c_dataengine PRIVATE gpu_queue
                                     ${CUDNN_PATH}/lib64/libcudnn.so
                                     ${CUDA_PATH}/lib64/libcudart.so
                                     ${CUDA_PATH}/lib64/stubs/libcuda.so)
endif ()

if (ENABLE_TDTQUE)
    target_link_libraries(_c_dataengine PRIVATE ${TSDCLIENT})
endif ()

add_dependencies(_c_dataengine _c_mindrecord)
target_link_libraries(_c_dataengine PRIVATE _c_mindrecord)

if (USE_GLOG)
    target_link_libraries(_c_dataengine PRIVATE mindspore::glog)
endif()

