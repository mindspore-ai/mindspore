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
