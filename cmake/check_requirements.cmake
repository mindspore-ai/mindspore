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
set(Python3_USE_STATIC_LIBS FALSE)
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


## packages used on Linux
if(NOT CMAKE_SYSTEM_NAME MATCHES "Windows")
    if(ENABLE_MINDDATA)
        find_required_program(tclsh)
    endif()

    if(MS_BUILD_GRPC)
        find_required_package(OpenSSL)
    endif()

    ## packages used in GPU mode only
    if(ENABLE_GPU)
        find_library(gmp_LIB gmp)
        find_library(gmpxx_LIB gmpxx)
        find_file(gmp_HEADER gmp.h)
        if(NOT gmp_LIB OR NOT gmpxx_LIB OR NOT gmp_HEADER)
            message(FATAL_ERROR "Required package gmp not found, please install gmp and try building MindSpore again.")
        endif()
        find_required_program(automake)
        find_required_program(autoconf)
        find_required_program(libtoolize)
        find_required_package(FLEX)
    endif()
endif()
