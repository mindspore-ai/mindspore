## define customized find functions, print customized error messages
function(find_required_package pkg_name)
    find_package(${pkg_name})
    if(NOT ${pkg_name}_FOUND)
        message(FATAL_ERROR "Required package ${pkg_name} not found, "
                "please install the package and try building MindSpore again.")
    endif()
endfunction()

function(find_required_program prog_name)
    find_program(${prog_name}_EXE ${prog_name})
    if(NOT ${prog_name}_EXE)
        message(FATAL_ERROR "Required program ${prog_name} not found, "
                "please install the package and try building MindSpore again.")
    endif()
endfunction()


## find python, quit if the found python is static
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(Python3_FIND_REGISTRY LAST)
  set(Python3_FIND_STRATEGY LOCATION)
endif()
set(Python3_USE_STATIC_LIBS FALSE)
set(Python3_FIND_VIRTUALENV ONLY)
find_package(Python3 COMPONENTS Interpreter Development)
if(Python3_FOUND)
    message("Python3 found, version: ${Python3_VERSION}")
    message("Python3 library path: ${Python3_LIBRARY}")
    message("Python3 interpreter: ${Python3_EXECUTABLE}")
elseif(Python3_LIBRARY AND Python3_EXECUTABLE AND
        ${Python3_VERSION} VERSION_GREATER_EQUAL "3.7.0" AND ${Python3_VERSION} VERSION_LESS "3.9.9")
    message(WARNING "Maybe python3 environment is broken.")
    message("Python3 library path: ${Python3_LIBRARY}")
    message("Python3 interpreter: ${Python3_EXECUTABLE}")
else()
    message(FATAL_ERROR "Python3 not found, please install Python>=3.7.5, and set --enable-shared "
            "if you are building Python locally")
endif()

## packages used both on windows and linux
if(DEFINED ENV{MS_PATCH_PATH})
    find_program(Patch_EXECUTABLE patch PATHS $ENV{MS_PATCH_PATH})
    set(Patch_FOUND ${Patch_EXECUTABLE})
else()
    find_package(Patch)
endif()
if(NOT Patch_FOUND)
    message(FATAL_ERROR "Patch not found, "
            "please set environment variable MS_PATCH_PATH to path where Patch is located, "
            "usually found in GIT_PATH/usr/bin on Windows")
endif()
message(PATCH_EXECUTABLE = ${Patch_EXECUTABLE})

find_required_package(Threads)

# add openmp if the onednn use ms threadpool
if(USE_MS_THREADPOOL_FOR_DNNL)
    find_package(OpenMP)
    if(OPENMP_FOUND)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
    else()
        message(WARNING "OpenMP not found")
    endif()
endif()

## packages used on Linux
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(ENABLE_MINDDATA)
        find_required_program(tclsh)
    endif()

    ## packages used in GPU mode only
    if(ENABLE_GPU)
        find_required_program(automake)
        find_required_program(autoconf)
        find_required_program(libtoolize)
        find_required_package(FLEX)
    endif()
endif()

# for macos, find appropriate macosx SDK then set SDKROOT and MACOSX_DEPLOYMENT_TARGET
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    if(NOT DEFINED ENV{SDKROOT})
        # arm64: macosx11.x
        # x86_64: macosx10.x, macosx11.x
        if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64")
            set(MACOSX_SDK_REGEX "MacOSX11(\\.\\d+)?")
        else()
            set(MACOSX_SDK_REGEX "MacOSX1[01](\\.\\d+)?")
        endif()
        set(MACOSX_XCODE_SDK_PATH "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs")
        set(MACOSX_CLT_SDK_PATH "/Library/Developer/CommandLineTools/SDKs")
        set(MACOSX_SDK_SEARCH_PATHS "${MACOSX_XCODE_SDK_PATH}/*" "${MACOSX_CLT_SDK_PATH}/*")
        file(GLOB ALL_SDK_NAME ${MACOSX_SDK_SEARCH_PATHS})
        # get highest SDK version meets the requirements
        execute_process(
            COMMAND bash -c "echo '${ALL_SDK_NAME}' | grep -Eo '${MACOSX_SDK_REGEX}' | sort -n | tail -1 | tr -d '\\n'"
            OUTPUT_VARIABLE MACOSX_FIND_SDK_NAME
        )
        if(NOT MACOSX_FIND_SDK_NAME)
            message(FATAL_ERROR
                "can not find appropriate macosx SDK, find in ${ALL_SDK_NAME}, you may set SDKROOT manually"
            )
        endif()
        if(IS_DIRECTORY "${MACOSX_XCODE_SDK_PATH}/${MACOSX_FIND_SDK_NAME}.sdk")
            set(CMAKE_OSX_SYSROOT "${MACOSX_XCODE_SDK_PATH}/${MACOSX_FIND_SDK_NAME}.sdk")
        else()
            set(CMAKE_OSX_SYSROOT "${MACOSX_CLT_SDK_PATH}/${MACOSX_FIND_SDK_NAME}.sdk")
        endif()
        set(ENV{SDKROOT} ${CMAKE_OSX_SYSROOT})
    endif()
    message("macosx sdkroot: $ENV{SDKROOT}")
    # set macosx deployment target based on SDK
    if(NOT DEFINED ENV{MACOSX_DEPLOYMENT_TARGET})
        execute_process(
            COMMAND bash -c "cat $ENV{SDKROOT}/SDKSettings.json | \
                grep -Eo 'MACOSX_DEPLOYMENT_TARGET\\\":\\\"\\d{2}\\.\\d+' | cut -d '\"' -f 3 | tr -d '\\n'"
            OUTPUT_VARIABLE MACOSX_FIND_SDK_VERSION
        )
        if(NOT MACOSX_FIND_SDK_VERSION)
            message(FATAL_ERROR "can not find MACOSX_DEPLOYMENT_TARGET in SDKROOT, \
                please check whether it's a valid SDK path")
        endif()

        if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64")
            set(CMAKE_OSX_DEPLOYMENT_TARGET "11.0")
        elseif(${MACOSX_FIND_SDK_VERSION} VERSION_LESS "10.15")
            set(CMAKE_OSX_DEPLOYMENT_TARGET ${MACOSX_FIND_SDK_VERSION} CACHE STRING
                "minimum macosx deployment target version" FORCE)
        else()
            set(CMAKE_OSX_DEPLOYMENT_TARGET "10.15")
        endif()
        set(ENV{MACOSX_DEPLOYMENT_TARGET} ${CMAKE_OSX_DEPLOYMENT_TARGET})
    endif()
    message("macosx deployment target version: $ENV{MACOSX_DEPLOYMENT_TARGET}")
endif()
