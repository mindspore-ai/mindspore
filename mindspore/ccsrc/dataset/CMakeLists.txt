set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sequence-point")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-variable")

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

############################# Options ################################
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_definitions(-D _CRT_RAND_S)
endif ()
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
add_subdirectory(text)
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
    $<TARGET_OBJECTS:engine-opt>
    $<TARGET_OBJECTS:engine>
    $<TARGET_OBJECTS:text>
    $<TARGET_OBJECTS:text-kernels>
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
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    target_link_libraries(_c_dataengine PRIVATE mindspore::pybind11_module ${PYTHON_LIBRARIES} mindspore::protobuf ${SECUREC_LIBRARY})
else()
    target_link_libraries(_c_dataengine PRIVATE mindspore::pybind11_module -ldl mindspore::protobuf ${SECUREC_LIBRARY})
endif()
target_link_libraries(_c_dataengine PUBLIC mindspore::jpeg_turbo mindspore::opencv_core mindspore::opencv_imgcodecs
        mindspore::opencv_imgproc mindspore::tinyxml2)
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
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(MINDRECORD_LINK_OBJECT ${CMAKE_BINARY_DIR}/mindspore/ccsrc/mindrecord/CMakeFiles/_c_mindrecord.dir/objects.a)
    target_link_libraries(_c_dataengine PRIVATE _c_mindrecord ${MINDRECORD_LINK_OBJECT} mindspore::sqlite)
else()
    target_link_libraries(_c_dataengine PRIVATE _c_mindrecord)
endif()

if (USE_GLOG)
    target_link_libraries(_c_dataengine PRIVATE mindspore::glog)
else()
    if (CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_options(_c_dataengine PRIVATE -Wl,-init,mindspore_log_init)
    elseif (CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set_target_properties(_c_dataengine PROPERTIES MACOSX_RPATH ON)
    endif ()
endif()
