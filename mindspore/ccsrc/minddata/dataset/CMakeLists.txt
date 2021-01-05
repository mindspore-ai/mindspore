set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized -Wno-deprecated-declarations
                                            -Wno-delete-non-abstract-non-virtual-dtor
                                            -Wno-constant-conversion -Wno-unknown-warning-option")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
endif()

# fix for GCC8.0
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-class-memaccess  -Wno-stringop-truncation")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")

############################# Options ################################
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_definitions(-D _CRT_RAND_S)
endif()
if(ENABLE_ACL)
    add_definitions(-D ENABLE_ACL)
    message(STATUS "ACL module is enabled")
endif()
if(ENABLE_GPUQUE)
    add_definitions(-D ENABLE_GPUQUE)
    message(STATUS "GPU queue is enabled")
endif()
if(ENABLE_TDTQUE)
    add_definitions(-D ENABLE_TDTQUE)
    message(STATUS "TDT queue is enabled")
endif()
if(MS_BUILD_GRPC)
    set(ENABLE_CACHE true)
    add_definitions(-D ENABLE_CACHE)
    message(STATUS "Cache is enabled")
endif()

# conde coverage
# option(ENABLE_COVERAGE "Enable code coverage report" OFF)
# if(ENABLE_COVERAGE)
#   include(${CMAKE_SOURCE_DIR}/cmake/CodeCoverage.cmake)
#   append_coverage_compiler_flags()
# endif()

########### Set up the include directories ###########################
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/runtime/device/ascend/platform)

include_directories(${CMAKE_BINARY_DIR}) # for protobuf generated .h

include_directories(${CMAKE_SOURCE_DIR}/mindspore)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/mindrecord/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/include)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image)

######################################################################

####################### Flags ########################################
# compile flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,-rpath,$ORIGIN:$ORIGIN/lib")
endif()

################## Include sub-modules ###############################
add_subdirectory(util)
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(engine)
add_subdirectory(api)
add_subdirectory(text)
add_subdirectory(callback)
######################################################################
add_dependencies(utils core)
add_dependencies(kernels-image core)
add_dependencies(kernels-data core)
add_dependencies(kernels-soft-dvpp-image core soft-dvpp-utils)
add_dependencies(kernels core)
add_dependencies(engine-datasetops-source core)
add_dependencies(engine-datasetops-source-sampler core)
add_dependencies(engine-datasetops core)
add_dependencies(engine-datasetops-mapop core)
add_dependencies(engine-opt core)
add_dependencies(engine-perf core)
add_dependencies(engine-gnn core)
add_dependencies(engine core)
add_dependencies(callback core)
add_dependencies(text core)
add_dependencies(text-kernels core)
add_dependencies(text-ir core)
add_dependencies(text-ir-kernels core)
add_dependencies(cpp-API core)
add_dependencies(engine-ir-datasetops core)
add_dependencies(engine-ir-datasetops-source core)
add_dependencies(engine-ir-datasetops-source-samplers core)
add_dependencies(engine-ir-cache core)
add_dependencies(kernels-ir core)
add_dependencies(kernels-ir-data core)
add_dependencies(kernels-ir-vision core)

if(ENABLE_ACL)
    add_dependencies(kernels-dvpp-image core dvpp-utils)
endif()

if(ENABLE_PYTHON)
    add_dependencies(APItoPython core)
endif()
if(ENABLE_TDTQUE)
    add_dependencies(engine-tdt core)
endif()
if(ENABLE_CACHE)
    add_dependencies(APItoPython engine-cache-client)
    add_dependencies(cpp-API engine-cache-client)
    add_dependencies(engine-ir-cache engine-cache-client)
    add_dependencies(engine-ir-datasetops engine-cache-client)
    add_dependencies(engine-opt engine-cache-client)
    add_dependencies(engine-datasetops engine-cache-client)
    add_dependencies(engine-perf engine-cache-client)
    add_dependencies(engine-cache-client core)
    add_dependencies(engine-cache-server core)
endif()
################### Create _c_dataengine Library ######################
set(submodules
        $<TARGET_OBJECTS:core>
        $<TARGET_OBJECTS:callback>
        $<TARGET_OBJECTS:utils>
        $<TARGET_OBJECTS:kernels>
        $<TARGET_OBJECTS:kernels-image>
        $<TARGET_OBJECTS:lite-cv>
        $<TARGET_OBJECTS:kernels-data>
        $<TARGET_OBJECTS:cpp-API>
        $<TARGET_OBJECTS:engine-ir-datasetops>
        $<TARGET_OBJECTS:engine-ir-datasetops-source>
        $<TARGET_OBJECTS:engine-ir-datasetops-source-samplers>
        $<TARGET_OBJECTS:engine-ir-cache>
        $<TARGET_OBJECTS:kernels-soft-dvpp-image>
        $<TARGET_OBJECTS:soft-dvpp-utils>
        $<TARGET_OBJECTS:engine-datasetops-source>
        $<TARGET_OBJECTS:engine-datasetops-source-sampler>
        $<TARGET_OBJECTS:engine-datasetops-mapop>
        $<TARGET_OBJECTS:engine-gnn>
        $<TARGET_OBJECTS:engine-perf>
        $<TARGET_OBJECTS:engine-datasetops>
        $<TARGET_OBJECTS:engine-opt>
        $<TARGET_OBJECTS:engine-cache-client>
        $<TARGET_OBJECTS:engine>
        $<TARGET_OBJECTS:text>
        $<TARGET_OBJECTS:text-kernels>
        $<TARGET_OBJECTS:text-ir>
        $<TARGET_OBJECTS:text-ir-kernels>
        $<TARGET_OBJECTS:kernels-ir>
        $<TARGET_OBJECTS:kernels-ir-data>
        $<TARGET_OBJECTS:kernels-ir-vision>
        )

if(ENABLE_ACL)
    set(submodules
            ${submodules}
            $<TARGET_OBJECTS:kernels-dvpp-image>
            $<TARGET_OBJECTS:dvpp-utils>)
endif()

if(ENABLE_PYTHON)
    set(submodules
        ${submodules}
        $<TARGET_OBJECTS:APItoPython>)
endif()

if(ENABLE_TDTQUE)
    add_library(_c_dataengine SHARED ${submodules} $<TARGET_OBJECTS:engine-tdt>)
else()
    add_library(_c_dataengine SHARED ${submodules})
endif()

if(ENABLE_PYTHON)
set_target_properties(_c_dataengine PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif()

######################################################################

################# Link with external libraries ########################
target_link_libraries(_c_dataengine PRIVATE mindspore mindspore_gvar)

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_MACOSX_RPATH 1)
    set(CMAKE_INSTALL_RPATH "@loader_path/lib;@loader_path")
    set_target_properties(_c_dataengine PROPERTIES INSTALL_RPATH "${CMAKE_INSTALL_RPATH}")
endif()

set(targetso $<TARGET_FILE:_c_dataengine>)
set(target_tiny "libtinyxml")
set(target_icuuc "libicuuc")
set(target_icudata "libicudata")
set(target_icui18n "libicui18n")
function(changerpath targetso linkso)
    set(some-file "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/${linkso}.txt")
    set(some-file1 "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/${linkso}1.txt")
    set(some-file2 "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/${linkso}2.txt")
    add_custom_command(
            OUTPUT
            ${some-file}
            COMMAND
            otool -L ${targetso} | grep ${linkso} > ${some-file}
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset"
            VERBATIM
    )
    add_custom_command(
            OUTPUT
            ${some-file1}
            COMMAND
            cat ${some-file} | cut -d " " -f 1 | sed -E "s/^.//g" > ${some-file1}
            DEPENDS
            ${some-file}
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset"
            VERBATIM
    )
    add_custom_command(
            OUTPUT
            ${some-file2}
            COMMAND
            awk -F "/"  "{print $NF}"  ${some-file1} > ${some-file2}
            DEPENDS
            ${some-file1}
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset"
            VERBATIM
    )
    add_custom_target(
            link_${linkso} ALL
            COMMAND install_name_tool -change `cat ${some-file1}` @rpath/`cat ${some-file2}` ${targetso}
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset"
            DEPENDS ${targetso} ${some-file1} ${some-file2}
            COMMENT "install tool name")
endfunction()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    changerpath(${targetso} ${target_tiny})
    changerpath(${targetso} ${target_icuuc})
    add_dependencies(link_${target_icuuc} link_${target_tiny})
    changerpath(${targetso} ${target_icudata})
    add_dependencies(link_${target_icudata} link_${target_icuuc})
    changerpath(${targetso} ${target_icui18n})
    add_dependencies(link_${target_icui18n} link_${target_icudata})
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    if(ENABLE_PYTHON)
        target_link_libraries(_c_dataengine PRIVATE mindspore::pybind11_module ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY})
    else()
        target_link_libraries(_c_dataengine PRIVATE ${SECUREC_LIBRARY})
    endif()
else()
    set(ICU_LIB mindspore::icuuc mindspore::icudata mindspore::icui18n)
    if(ENABLE_PYTHON)
        target_link_libraries(_c_dataengine PRIVATE mindspore::pybind11_module -ldl ${SECUREC_LIBRARY})
    else()
        target_link_libraries(_c_dataengine PRIVATE -ldl ${SECUREC_LIBRARY})
    endif()
    target_link_libraries(_c_dataengine PUBLIC mindspore::sentencepiece)
endif()

target_link_libraries(_c_dataengine PUBLIC mindspore::jpeg_turbo mindspore::turbojpeg mindspore::opencv_core
                                          mindspore::opencv_imgcodecs mindspore::opencv_imgproc mindspore::tinyxml2
                                          mindspore::sentencepiece_train ${ICU_LIB})
if(ENABLE_GPUQUE)
    target_link_libraries(_c_dataengine PRIVATE gpu_queue
                          ${CUDNN_LIBRARY_PATH}
                          ${CUDA_PATH}/lib64/libcudart.so
                          ${CUDA_PATH}/lib64/stubs/libcuda.so)
endif()

if(ENABLE_TDTQUE)
    target_link_libraries(_c_dataengine PRIVATE ${ACL})
endif()

add_dependencies(_c_dataengine _c_mindrecord)
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(MINDRECORD_LINK_OBJECT
        ${CMAKE_BINARY_DIR}/mindspore/ccsrc/minddata/mindrecord/CMakeFiles/_c_mindrecord.dir/objects.a)
    target_link_libraries(_c_dataengine PRIVATE _c_mindrecord ${MINDRECORD_LINK_OBJECT} mindspore::sqlite)
else()
    target_link_libraries(_c_dataengine PRIVATE _c_mindrecord)
    if(ENABLE_CPU AND (ENABLE_D OR ENABLE_GPU))
        if(${ENABLE_IBVERBS} STREQUAL "ON")
            target_link_libraries(_c_dataengine PRIVATE ibverbs rdmacm)
        endif()
        target_link_libraries(_c_dataengine PRIVATE ps_cache)
    endif()
endif()

target_link_libraries(_c_dataengine PRIVATE mindspore_core mindspore_shared_lib)

if(USE_GLOG)
    target_link_libraries(_c_dataengine PRIVATE mindspore::glog)
else()
    if(CMAKE_SYSTEM_NAME MATCHES "Linux")
        target_link_options(_c_dataengine PRIVATE -Wl,-init,mindspore_log_init)
    elseif(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        set_target_properties(_c_dataengine PROPERTIES MACOSX_RPATH ON)
    endif()
endif()

if(MS_BUILD_GRPC)
    target_link_libraries(_c_dataengine PRIVATE mindspore::grpc++)
endif()

set_target_properties(_c_dataengine PROPERTIES INSTALL_RPATH ${MINDSPORE_RPATH})
