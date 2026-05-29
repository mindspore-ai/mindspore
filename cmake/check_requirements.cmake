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


## find python — follow the active shell context (conda/venv) and verify the
## packaging stack at configure time, so a misrouted interpreter fails fast
## instead of after a multi-hour build.
if(CMAKE_SYSTEM_NAME MATCHES "Windows")
  set(Python3_FIND_REGISTRY LAST)
  set(Python3_FIND_STRATEGY LOCATION)
endif()
set(Python3_USE_STATIC_LIBS FALSE)

# Lock onto the shell's active environment unless the caller passed
# -DPython3_EXECUTABLE explicitly. CMake's Python3_FIND_VIRTUALENV does not
# reliably honor CONDA_PREFIX, which is what causes /usr/local/pythonX.Y to be
# selected over an active conda env on multi-Python machines.
if(NOT DEFINED Python3_EXECUTABLE)
    if(DEFINED ENV{CONDA_PREFIX})
        set(Python3_ROOT_DIR "$ENV{CONDA_PREFIX}" CACHE PATH "Active conda env" FORCE)
        set(Python3_EXECUTABLE "$ENV{CONDA_PREFIX}/bin/python" CACHE FILEPATH "Conda python" FORCE)
    elseif(DEFINED ENV{VIRTUAL_ENV})
        set(Python3_ROOT_DIR "$ENV{VIRTUAL_ENV}" CACHE PATH "Active virtualenv" FORCE)
        set(Python3_EXECUTABLE "$ENV{VIRTUAL_ENV}/bin/python" CACHE FILEPATH "Venv python" FORCE)
    endif()
endif()
set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_STRATEGY LOCATION)

find_package(Python3 COMPONENTS Interpreter Development)
if(NOT Python3_FOUND)
    message(FATAL_ERROR "Python3 not found. Activate a conda env or venv with Python "
            "3.9-3.12 before running build.sh.")
endif()
if(${Python3_VERSION} VERSION_LESS "3.9.0" OR ${Python3_VERSION} VERSION_GREATER_EQUAL "3.13.0")
    message(FATAL_ERROR "Found Python ${Python3_VERSION} at ${Python3_EXECUTABLE}, "
            "but MindSpore requires 3.9-3.12 (numpy>=1.25 floor introduced in master).")
endif()
message(STATUS "Python3 ${Python3_VERSION} at ${Python3_EXECUTABLE}")
message(STATUS "Python3 library: ${Python3_LIBRARY}")

# Verify the packaging stack up front. Without this check, a missing 'wheel'
# in the chosen interpreter only surfaces as "invalid command 'bdist_wheel'"
# at the end of the build, after CMake/compile have already succeeded.
foreach(_pkg setuptools wheel)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import ${_pkg}; print(${_pkg}.__version__)"
        RESULT_VARIABLE _rc
        OUTPUT_VARIABLE _ver
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    if(NOT _rc EQUAL 0)
        message(FATAL_ERROR "Python at ${Python3_EXECUTABLE} cannot import '${_pkg}'. "
                "Install with: ${Python3_EXECUTABLE} -m pip install ${_pkg}")
    endif()
    message(STATUS "  ${_pkg}: ${_ver}")
endforeach()
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "from wheel.bdist_wheel import bdist_wheel"
    RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
if(NOT _rc EQUAL 0)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "from setuptools.command.bdist_wheel import bdist_wheel"
        RESULT_VARIABLE _rc OUTPUT_QUIET ERROR_QUIET)
endif()
if(NOT _rc EQUAL 0)
    message(FATAL_ERROR "bdist_wheel command not available in ${Python3_EXECUTABLE}. "
            "Install: ${Python3_EXECUTABLE} -m pip install 'wheel<0.45' 'setuptools<70'")
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
    # compiler `clang` on Darwin does not support option `-fopenmp`
    if(OPENMP_FOUND AND (NOT CMAKE_SYSTEM_NAME MATCHES "Darwin"))
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
        # arm64: macosx11.x and above (e.g. 11.x, 12.x, ..., 26.x, ...), actively maintained
        # x86_64: macosx10.x or macosx11.x only, no longer maintained
        if(${CMAKE_HOST_SYSTEM_PROCESSOR} MATCHES "arm64")
            set(MACOSX_SDK_REGEX "MacOSX[1-9][0-9]*(\\.\\d+)?")
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
