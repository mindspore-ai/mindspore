if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-sign-compare")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-reorder")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-switch")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-uninitialized -Wno-deprecated-declarations \
                                            -Wno-delete-non-abstract-non-virtual-dtor       \
                                            -Wno-unused-comparison -Wno-constant-conversion \
                                            -Wno-unknown-warning-option -Wno-macro-redefined")
else()
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-maybe-uninitialized")
    endif()
endif()

if(MSVC)
    # define this for msvc:dllexport or dllimport
    add_compile_definitions(BUILDING_DATASET_DLL)
endif()

# fix for GCC8.0
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-class-memaccess  -Wno-stringop-truncation")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-attributes")
endif()

############################# Options ################################
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    add_definitions(-D _CRT_RAND_S)
endif()
if(ENABLE_ACL)
    add_definitions(-D ENABLE_ACL)
    message(STATUS "ACL module is enabled")
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
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset)
include_directories(${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/kernels/image)


######################################################################

####################### Flags ########################################
# compile flags
if(NOT MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fvisibility=default")
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-delete-abstract-non-virtual-dtor")
endif()

set(MINDDATA_LOG_ADAPTER_SRC ${CMAKE_SOURCE_DIR}/mindspore/ccsrc/minddata/dataset/util/md_log_adapter.cc)
add_library(md_log_adapter_obj OBJECT ${MINDDATA_LOG_ADAPTER_SRC})
add_library(md_log_adapter STATIC $<TARGET_OBJECTS:md_log_adapter_obj>)
target_link_libraries(md_log_adapter mindspore_core)

################## Include sub-modules ###############################
add_subdirectory(util)
add_subdirectory(core)
add_subdirectory(kernels)
add_subdirectory(engine)
add_subdirectory(api)
add_subdirectory(audio)
add_subdirectory(text)
add_subdirectory(callback)
add_subdirectory(plugin)
######################################################################
add_dependencies(utils core)
add_dependencies(kernels-image core)
add_dependencies(kernels-data core)
add_dependencies(kernels core)
add_dependencies(engine-datasetops-source core)
add_dependencies(engine-datasetops-source-sampler core)
add_dependencies(engine-datasetops core)
add_dependencies(engine-datasetops-mapop core)
add_dependencies(engine-opt core)
add_dependencies(engine-gnn core)
add_dependencies(engine core)
add_dependencies(callback core)
add_dependencies(audio-kernels core)
add_dependencies(audio-ir core)
add_dependencies(audio-ir-kernels core)
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
add_dependencies(kernels-dvpp-image core)

if(ENABLE_PYTHON)
    add_dependencies(APItoPython core)
    add_dependencies(APItoPython engine-gnn)
endif()
if(NOT ENABLE_SECURITY)
    add_dependencies(engine-perf core)
endif()

# for proto/cache_grpc.pb.h dependency
if(ENABLE_CACHE)
    add_dependencies(APItoPython engine-cache-server)
    add_dependencies(cpp-API engine-cache-server)
    add_dependencies(engine-ir-cache engine-cache-server)
    add_dependencies(engine-ir-datasetops engine-cache-server)
    add_dependencies(engine-ir-datasetops-source engine-cache-server)
    add_dependencies(engine-opt engine-cache-server)
    add_dependencies(engine-datasetops engine-cache-server)
    add_dependencies(engine-datasetops-source engine-cache-server)
    if(NOT ENABLE_SECURITY)
        add_dependencies(engine-perf engine-cache-server)
    endif()
    add_dependencies(engine-cache-client core)
    add_dependencies(engine-cache-server core)
else()
    add_dependencies(APItoPython engine-cache-client)
    add_dependencies(cpp-API engine-cache-client)
    add_dependencies(engine-ir-cache engine-cache-client)
    add_dependencies(engine-ir-datasetops engine-cache-client)
    add_dependencies(engine-ir-datasetops-source engine-cache-client)
    add_dependencies(engine-opt engine-cache-client)
    add_dependencies(engine-datasetops engine-cache-client)
    add_dependencies(engine-datasetops-source engine-cache-client)
    if(NOT ENABLE_SECURITY)
        add_dependencies(engine-perf engine-cache-client)
    endif()
    add_dependencies(engine-cache-client core)
endif()

################### Create _c_dataengine Library ######################
set(dataengine_submodules
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
        $<TARGET_OBJECTS:engine-datasetops-source>
        $<TARGET_OBJECTS:engine-datasetops-source-sampler>
        $<TARGET_OBJECTS:engine-datasetops-mapop>
        $<TARGET_OBJECTS:engine-gnn>
        $<TARGET_OBJECTS:engine-datasetops>
        $<TARGET_OBJECTS:engine-opt>
        $<TARGET_OBJECTS:engine-cache-client>
        $<TARGET_OBJECTS:engine>
        $<TARGET_OBJECTS:audio-kernels>
        $<TARGET_OBJECTS:audio-ir>
        $<TARGET_OBJECTS:audio-ir-kernels>
        $<TARGET_OBJECTS:text>
        $<TARGET_OBJECTS:text-kernels>
        $<TARGET_OBJECTS:text-ir>
        $<TARGET_OBJECTS:text-ir-kernels>
        $<TARGET_OBJECTS:kernels-ir>
        $<TARGET_OBJECTS:kernels-ir-data>
        $<TARGET_OBJECTS:kernels-ir-vision>
        $<TARGET_OBJECTS:md_plugin>
        $<TARGET_OBJECTS:kernels-dvpp-image>)

if(ENABLE_PYTHON)
    set(dataengine_submodules
        ${dataengine_submodules}
        $<TARGET_OBJECTS:APItoPython>)
endif()

if(NOT ENABLE_SECURITY)
    set(dataengine_submodules ${dataengine_submodules} $<TARGET_OBJECTS:engine-perf>)
endif()

set(dataengine_submodules ${dataengine_submodules} CACHE INTERNAL "_c_dataengine objects")
add_library(_c_dataengine SHARED ${dataengine_submodules})
add_dependencies(_c_dataengine md_log_adapter)
target_link_libraries(_c_dataengine PRIVATE md_log_adapter)
if(ENABLE_PYTHON)
set_target_properties(_c_dataengine PROPERTIES
    PREFIX "${PYTHON_MODULE_PREFIX}"
    SUFFIX "${PYTHON_MODULE_EXTENSION}"
    )
endif()

######################################################################

################# Link with external libraries ########################
if(NOT MSLITE_ENABLE_CLOUD_MIND_DATA)
    target_link_libraries(_c_dataengine PRIVATE mindspore)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    if(ENABLE_PYTHON)
        target_link_libraries(_c_dataengine PRIVATE mindspore::pybind11_module ${PYTHON_LIBRARIES} ${SECUREC_LIBRARY})
    else()
        target_link_libraries(_c_dataengine PRIVATE ${SECUREC_LIBRARY})
    endif()
    if(NOT MSVC)
        target_link_options(_c_dataengine PRIVATE -Wl,--allow-multiple-definition)
    endif()
else()
    set(ICU_LIB mindspore::icuuc mindspore::icudata mindspore::icui18n)
    if(ENABLE_PYTHON)
        target_link_libraries(_c_dataengine PRIVATE mindspore::pybind11_module -ldl ${SECUREC_LIBRARY})
    else()
        target_link_libraries(_c_dataengine PRIVATE -ldl ${SECUREC_LIBRARY})
    endif()
    target_link_libraries(_c_dataengine PUBLIC mindspore::sentencepiece mindspore::z)
endif()

target_link_libraries(_c_dataengine PUBLIC mindspore::jpeg_turbo mindspore::turbojpeg mindspore::opencv_core
                                          mindspore::opencv_imgcodecs mindspore::opencv_imgproc mindspore::tinyxml2
                                          mindspore::sentencepiece_train ${ICU_LIB})

target_link_libraries(_c_dataengine PRIVATE mindspore_backend)

add_dependencies(_c_dataengine _c_mindrecord)
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    if(MSVC)
        target_link_libraries(_c_dataengine PRIVATE _c_mindrecord mindspore::sqlite)
    else()
        set(MINDRECORD_LINK_OBJECT
            ${CMAKE_BINARY_DIR}/mindspore/ccsrc/minddata/mindrecord/CMakeFiles/_c_mindrecord.dir/objects.a)
        target_link_libraries(_c_dataengine PRIVATE _c_mindrecord ${MINDRECORD_LINK_OBJECT} mindspore::sqlite)
    endif()
else()
    target_link_libraries(_c_dataengine PRIVATE _c_mindrecord)
    if(ENABLE_CPU AND NOT WIN32)
        target_link_libraries(_c_dataengine PRIVATE ps_cache)
    endif()
endif()

if(MSLITE_ENABLE_CLOUD_MIND_DATA)
    target_link_libraries(_c_dataengine PRIVATE mindspore_shared_lib mindspore::grpc++)
    target_link_libraries(_c_dataengine PUBLIC mindspore::protobuf)
else()
    target_link_libraries(_c_dataengine PRIVATE mindspore_core)
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(_c_dataengine PRIVATE mindspore_shared_lib)
endif()

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
    if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
        target_link_libraries(_c_dataengine PRIVATE -Wl,--no-as-needed mindspore::grpc++)
    else()
        target_link_libraries(_c_dataengine PRIVATE mindspore::grpc++)
    endif()
endif()

if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin" AND NOT MSLITE_ENABLE_CLOUD_MIND_DATA)
    set_target_properties(_c_dataengine PROPERTIES INSTALL_RPATH
        ${ORIGIN_PATH}:${ORIGIN_PATH}/lib:${ORIGIN_PATH}/lib/plugin)
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    include(${CMAKE_SOURCE_DIR}/cmake/change_rpath.cmake)
    set(LINK_SOS libtinyxml libicuuc libicudata libicui18n)
    changerpath($<TARGET_FILE:_c_dataengine> _c_dataengine "${LINK_SOS}")
endif()
