set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)

set(USED_CMAKE_GENERATOR "${CMAKE_GENERATOR}" CACHE STRING "Expose CMAKE_GENERATOR" FORCE)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.18.3")
    set(MODERN_CMAKE YES)
    message(STATUS "Merging integrated CMake 3.18.3+ iOS toolchain(s) with this toolchain!")
endif()

# Get the Xcode version
execute_process(COMMAND xcodebuild -version OUTPUT_VARIABLE XCODE_VERSION ERROR_QUIET OUTPUT_STRIP_TRAILING_WHITESPACE)
string(REGEX MATCH "Xcode [0-9\\.]+" XCODE_VERSION "${XCODE_VERSION}")
string(REGEX REPLACE "Xcode ([0-9\\.]+)" "\\1" XCODE_VERSION "${XCODE_VERSION}")

set(FORCE_CACHE FORCE)
get_property(_CMAKE_IN_TRY_COMPILE GLOBAL PROPERTY IN_TRY_COMPILE)
if(_CMAKE_IN_TRY_COMPILE)
    unset(FORCE_CACHE)
endif()

if(NOT DEFINED PLATFORM)
    if(CMAKE_OSX_ARCHITECTURES)
        if(CMAKE_OSX_ARCHITECTURES MATCHES ".*arm.*" AND CMAKE_OSX_SYSROOT MATCHES ".*iphoneos.*")
            set(PLATFORM "OS")
        elseif(CMAKE_OSX_ARCHITECTURES MATCHES "i386" AND CMAKE_OSX_SYSROOT MATCHES ".*iphonesimulator.*")
            set(PLATFORM "SIMULATOR")
        endif()
    endif()
    if(NOT PLATFORM)
        set(PLATFORM "OS")
    endif()
endif()

set(PLATFORM_INT "${PLATFORM}" CACHE STRING "Type of platform for which the build targets.")

if(PLATFORM_INT STREQUAL "OS" AND DEPLOYMENT_TARGET VERSION_GREATER_EQUAL 10.3.4)
    set(PLATFORM_INT "OS64")
    message(STATUS "Targeting minimum SDK version ${DEPLOYMENT_TARGET}. Dropping 32-bit support.")
elseif(PLATFORM_INT STREQUAL "SIMULATOR" AND DEPLOYMENT_TARGET VERSION_GREATER_EQUAL 10.3.4)
    set(PLATFORM_INT "SIMULATOR64")
    message(STATUS "Targeting minimum SDK version ${DEPLOYMENT_TARGET}. Dropping 32-bit support.")
endif()

if(PLATFORM_INT STREQUAL "OS")
    set(SDK_NAME iphoneos)
    if(NOT ARCHS)
        set(ARCHS armv7 armv7s arm64)
    endif()
elseif(PLATFORM_INT STREQUAL "OS64")
    set(SDK_NAME iphoneos)
    if(NOT ARCHS)
        if(XCODE_VERSION VERSION_GREATER 10.0)
            set(ARCHS arm64)
        else()
            set(ARCHS arm64)
        endif()
    endif()
elseif(PLATFORM_INT STREQUAL "OS64COMBINED")
    set(SDK_NAME iphoneos)
    if(MODERN_CMAKE)
        if(NOT ARCHS)
            if(XCODE_VERSION VERSION_GREATER 10.0)
                set(ARCHS arm64 x86_64)
            else()
                set(ARCHS arm64 x86_64)
            endif()
        endif()
    else()
        message(FATAL_ERROR "Please make sure that you are running CMake 3.18.3+ to make the OS64COMBINED setting work")
    endif()
elseif(PLATFORM_INT STREQUAL "SIMULATOR")
    set(SDK_NAME iphonesimulator)
    if(NOT ARCHS)
        set(ARCHS i386)
    endif()
    message(DEPRECATION "SIMULATOR IS DEPRECATED. Consider using SIMULATOR64 instead.")
elseif(PLATFORM_INT STREQUAL "SIMULATOR64")
    set(SDK_NAME iphonesimulator)
    if(NOT ARCHS)
        set(ARCHS x86_64)
    endif()
else()
    message(FATAL_ERROR "Invalid PLATFORM: ${PLATFORM_INT}")
endif()
message(STATUS "Configuring ${SDK_NAME} build for platform: ${PLATFORM_INT}, architecture(s): ${ARCHS}")

if(MODERN_CMAKE AND PLATFORM_INT MATCHES ".*COMBINED" AND NOT USED_CMAKE_GENERATOR MATCHES "Xcode")
    message(FATAL_ERROR "The COMBINED options only work with Xcode generator, -G Xcode")
endif()

execute_process(COMMAND xcodebuild -version -sdk ${SDK_NAME} Path
        OUTPUT_VARIABLE CMAKE_OSX_SYSROOT_INT
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
if(NOT DEFINED CMAKE_OSX_SYSROOT_INT AND NOT DEFINED CMAKE_OSX_SYSROOT)
    message(SEND_ERROR "Please make sure that Xcode is installed and that the toolchain"
            "is pointing to the correct path. Please run:"
            "sudo xcode-select -s /Applications/Xcode.app/Contents/Developer"
            "and see if that fixes the problem for you.")
    message(FATAL_ERROR "Invalid CMAKE_OSX_SYSROOT: ${CMAKE_OSX_SYSROOT} "
            "does not exist.")
elseif(DEFINED CMAKE_OSX_SYSROOT)
    message(STATUS "Using SDK: ${CMAKE_OSX_SYSROOT} for platform: ${PLATFORM_INT} when checking compatibility")
elseif(DEFINED CMAKE_OSX_SYSROOT_INT)
    message(STATUS "Using SDK: ${CMAKE_OSX_SYSROOT_INT} for platform: ${PLATFORM_INT}")
    set(CMAKE_OSX_SYSROOT "${CMAKE_OSX_SYSROOT_INT}" CACHE INTERNAL "")
endif()

if(USED_CMAKE_GENERATOR MATCHES "Xcode")
    set(CMAKE_OSX_SYSROOT "${SDK_NAME}" CACHE INTERNAL "")
endif()

if(NOT DEFINED DEPLOYMENT_TARGET)
    set(DEPLOYMENT_TARGET "9.0" CACHE STRING "Minimum SDK version to build for.")
    message(STATUS "Using the default min-version since DEPLOYMENT_TARGET not provided!")
endif()
if(NOT DEFINED ENABLE_BITCODE AND NOT ARCHS MATCHES "((^|;|, )(i386|x86_64))+")
    message(STATUS "Enabling bitcode support by default. ENABLE_BITCODE not provided!")
    set(ENABLE_BITCODE TRUE)
elseif(NOT DEFINED ENABLE_BITCODE)
    message(STATUS "Disabling bitcode support by default on simulators. ENABLE_BITCODE not provided for override!")
    set(ENABLE_BITCODE FALSE)
endif()
set(ENABLE_BITCODE_INT ${ENABLE_BITCODE} CACHE BOOL "Whether or not to enable bitcode" ${FORCE_CACHE})
if(NOT DEFINED ENABLE_ARC)
    set(ENABLE_ARC TRUE)
    message(STATUS "Enabling ARC support by default. ENABLE_ARC not provided!")
endif()
set(ENABLE_ARC_INT ${ENABLE_ARC} CACHE BOOL "Whether or not to enable ARC" ${FORCE_CACHE})
if(NOT DEFINED ENABLE_VISIBILITY)
    set(ENABLE_VISIBILITY FALSE)
    message(STATUS "Hiding symbols visibility by default. ENABLE_VISIBILITY not provided!")
endif()
set(ENABLE_VISIBILITY_INT ${ENABLE_VISIBILITY} CACHE BOOL
        "Whether or not to hide symbols (-fvisibility=hidden)" ${FORCE_CACHE})
# Set strict compiler checks or not
if(NOT DEFINED ENABLE_STRICT_TRY_COMPILE)
    set(ENABLE_STRICT_TRY_COMPILE FALSE)
    message(STATUS "Using NON-strict compiler checks by default. ENABLE_STRICT_TRY_COMPILE not provided!")
endif()
set(ENABLE_STRICT_TRY_COMPILE_INT ${ENABLE_STRICT_TRY_COMPILE} CACHE BOOL
        "Whether or not to use strict compiler checks" ${FORCE_CACHE})
# Get the SDK version information.
execute_process(COMMAND xcodebuild -sdk ${CMAKE_OSX_SYSROOT} -version SDKVersion
        OUTPUT_VARIABLE SDK_VERSION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT DEFINED CMAKE_DEVELOPER_ROOT AND NOT USED_CMAKE_GENERATOR MATCHES "Xcode")
    get_filename_component(PLATFORM_SDK_DIR ${CMAKE_OSX_SYSROOT} PATH)
    get_filename_component(CMAKE_DEVELOPER_ROOT ${PLATFORM_SDK_DIR} PATH)

    if(NOT DEFINED CMAKE_DEVELOPER_ROOT)
        message(FATAL_ERROR "Invalid CMAKE_DEVELOPER_ROOT: "
                "${CMAKE_DEVELOPER_ROOT} does not exist.")
    endif()
endif()
if(NOT CMAKE_C_COMPILER)
    execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find clang
            OUTPUT_VARIABLE CMAKE_C_COMPILER
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Using C compiler: ${CMAKE_C_COMPILER}")
endif()
if(NOT CMAKE_CXX_COMPILER)
    execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find clang++
            OUTPUT_VARIABLE CMAKE_CXX_COMPILER
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "Using CXX compiler: ${CMAKE_CXX_COMPILER}")
endif()
execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find libtool
        OUTPUT_VARIABLE BUILD_LIBTOOL
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "Using libtool: ${BUILD_LIBTOOL}")


set(CMAKE_C_CREATE_STATIC_LIBRARY
        "${BUILD_LIBTOOL} -static -o <TARGET> <LINK_FLAGS> <OBJECTS> ")
set(CMAKE_CXX_CREATE_STATIC_LIBRARY
        "${BUILD_LIBTOOL} -static -o <TARGET> <LINK_FLAGS> <OBJECTS> ")

if(NOT CMAKE_INSTALL_NAME_TOOL)
    execute_process(COMMAND xcrun -sdk ${CMAKE_OSX_SYSROOT} -find install_name_tool
            OUTPUT_VARIABLE CMAKE_INSTALL_NAME_TOOL_INT
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CMAKE_INSTALL_NAME_TOOL ${CMAKE_INSTALL_NAME_TOOL_INT} CACHE STRING "" ${FORCE_CACHE})
    message(STATUS "Using install_name_tool: ${CMAKE_INSTALL_NAME_TOOL}")
endif()

execute_process(COMMAND uname -r
        OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_VERSION
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE)

if(MODERN_CMAKE)
    if(SDK_NAME MATCHES "iphone")
        set(CMAKE_SYSTEM_NAME iOS CACHE INTERNAL "" ${FORCE_CACHE})
    endif()

    if(PLATFORM_INT MATCHES ".*COMBINED")
        set(CMAKE_XCODE_ATTRIBUTE_ONLY_ACTIVE_ARCH NO CACHE INTERNAL "" ${FORCE_CACHE})
        set(CMAKE_IOS_INSTALL_COMBINED YES CACHE INTERNAL "" ${FORCE_CACHE})
        message(STATUS "Will combine built (static) artifacts into FAT lib...")
    endif()
else()
    set(CMAKE_SYSTEM_NAME Darwin CACHE INTERNAL "" ${FORCE_CACHE})
endif()
set(CMAKE_SYSTEM_VERSION ${SDK_VERSION} CACHE INTERNAL "")
set(UNIX TRUE CACHE BOOL "")
set(APPLE TRUE CACHE BOOL "")
set(IOS TRUE CACHE BOOL "")
set(CMAKE_AR ar CACHE FILEPATH "" FORCE)
set(CMAKE_RANLIB ranlib CACHE FILEPATH "" FORCE)
set(CMAKE_STRIP strip CACHE FILEPATH "" FORCE)
set(CMAKE_OSX_ARCHITECTURES ${ARCHS} CACHE STRING "Build architecture for iOS")
if(ENABLE_STRICT_TRY_COMPILE_INT)
    message(STATUS "Using strict compiler checks (default in CMake).")
else()
    set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
endif()
set(CMAKE_SHARED_LIBRARY_PREFIX "lib")
set(CMAKE_SHARED_LIBRARY_SUFFIX ".dylib")
set(CMAKE_SHARED_MODULE_PREFIX "lib")
set(CMAKE_SHARED_MODULE_SUFFIX ".so")
set(CMAKE_C_COMPILER_ABI ELF)
set(CMAKE_CXX_COMPILER_ABI ELF)
set(CMAKE_C_HAS_ISYSROOT 1)
set(CMAKE_CXX_HAS_ISYSROOT 1)
set(CMAKE_MODULE_EXISTS 1)
set(CMAKE_DL_LIBS "")
set(CMAKE_C_OSX_COMPATIBILITY_VERSION_FLAG "-compatibility_version ")
set(CMAKE_C_OSX_CURRENT_VERSION_FLAG "-current_version ")
set(CMAKE_CXX_OSX_COMPATIBILITY_VERSION_FLAG "${CMAKE_C_OSX_COMPATIBILITY_VERSION_FLAG}")
set(CMAKE_CXX_OSX_CURRENT_VERSION_FLAG "${CMAKE_C_OSX_CURRENT_VERSION_FLAG}")

if(ARCHS MATCHES "((^|;|, )(arm64|arm64e|x86_64))+")
    set(CMAKE_C_SIZEOF_DATA_PTR 8)
    set(CMAKE_CXX_SIZEOF_DATA_PTR 8)
    if(ARCHS MATCHES "((^|;|, )(arm64|arm64e))+")
        set(CMAKE_SYSTEM_PROCESSOR "aarch64")
    else()
        set(CMAKE_SYSTEM_PROCESSOR "x86_64")
    endif()
    message(STATUS "Using a data_ptr size of 8")
else()
    set(CMAKE_C_SIZEOF_DATA_PTR 4)
    set(CMAKE_CXX_SIZEOF_DATA_PTR 4)
    set(CMAKE_SYSTEM_PROCESSOR "arm")
    message(STATUS "Using a data_ptr size of 4")
endif()

message(STATUS "Building for minimum ${SDK_NAME} version: ${DEPLOYMENT_TARGET} (SDK version: ${SDK_VERSION})")

if(PLATFORM_INT STREQUAL "OS" OR PLATFORM_INT STREQUAL "OS64")
    if(XCODE_VERSION VERSION_LESS 7.0)
        set(SDK_NAME_VERSION_FLAGS "-mios-version-min=${DEPLOYMENT_TARGET}")
    else()
        set(SDK_NAME_VERSION_FLAGS "-m${SDK_NAME}-version-min=${DEPLOYMENT_TARGET}")
    endif()
else()
    set(SDK_NAME_VERSION_FLAGS "-mios-simulator-version-min=${DEPLOYMENT_TARGET}")
endif()
message(STATUS "Version flags set to: ${SDK_NAME_VERSION_FLAGS}")
set(CMAKE_OSX_DEPLOYMENT_TARGET ${DEPLOYMENT_TARGET} CACHE STRING "Set CMake deployment target" ${FORCE_CACHE})

if(ENABLE_BITCODE_INT)
    set(BITCODE "-fembed-bitcode")
    set(CMAKE_XCODE_ATTRIBUTE_BITCODE_GENERATION_MODE bitcode CACHE INTERNAL "")
    message(STATUS "Enabling bitcode support.")
else()
    set(BITCODE "")
    set(CMAKE_XCODE_ATTRIBUTE_ENABLE_BITCODE NO CACHE INTERNAL "")
    message(STATUS "Disabling bitcode support.")
endif()

if(ENABLE_ARC_INT)
    set(FOBJC_ARC "-fobjc-arc")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC YES CACHE INTERNAL "")
    message(STATUS "Enabling ARC support.")
else()
    set(FOBJC_ARC "-fno-objc-arc")
    set(CMAKE_XCODE_ATTRIBUTE_CLANG_ENABLE_OBJC_ARC NO CACHE INTERNAL "")
    message(STATUS "Disabling ARC support.")
endif()

if(NOT ENABLE_VISIBILITY_INT)
    set(VISIBILITY "-fvisibility=hidden")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN YES CACHE INTERNAL "")
    message(STATUS "Hiding symbols (-fvisibility=hidden).")
else()
    set(VISIBILITY "")
    set(CMAKE_XCODE_ATTRIBUTE_GCC_SYMBOLS_PRIVATE_EXTERN NO CACHE INTERNAL "")
endif()

if(USED_CMAKE_GENERATOR MATCHES "Xcode")
    message(STATUS "Not setting any manual command-line buildflags, since Xcode is selected as generator.")
else()
    set(CMAKE_C_FLAGS "${SDK_NAME_VERSION_FLAGS} ${BITCODE} -fobjc-abi-version=2 ${FOBJC_ARC} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS
            "${SDK_NAME_VERSION_FLAGS} ${BITCODE} ${VISIBILITY}
            -fvisibility-inlines-hidden -fobjc-abi-version=2 ${FOBJC_ARC} ${CMAKE_CXX_FLAGS}")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -O0 -g ${CMAKE_CXX_FLAGS_DEBUG}")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS} -DNDEBUG -Os -ffast-math ${CMAKE_CXX_FLAGS_MINSIZEREL}")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO
            "${CMAKE_CXX_FLAGS} -DNDEBUG -O2 -g -ffast-math ${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -DNDEBUG -O3 -ffast-math ${CMAKE_CXX_FLAGS_RELEASE}")
    set(CMAKE_C_LINK_FLAGS "${SDK_NAME_VERSION_FLAGS} -Wl,-search_paths_first ${CMAKE_C_LINK_FLAGS}")
    set(CMAKE_CXX_LINK_FLAGS "${SDK_NAME_VERSION_FLAGS}  -Wl,-search_paths_first ${CMAKE_CXX_LINK_FLAGS}")

    list(APPEND VARS_TO_FORCE_IN_CACHE
            CMAKE_C_FLAGS
            CMAKE_CXX_FLAGS
            CMAKE_CXX_FLAGS_DEBUG
            CMAKE_CXX_FLAGS_RELWITHDEBINFO
            CMAKE_CXX_FLAGS_MINSIZEREL
            CMAKE_CXX_FLAGS_RELEASE
            CMAKE_C_LINK_FLAGS
            CMAKE_CXX_LINK_FLAGS)
    foreach(VAR_TO_FORCE ${VARS_TO_FORCE_IN_CACHE})
        set(${VAR_TO_FORCE} "${${VAR_TO_FORCE}}" CACHE STRING "")
    endforeach()
endif()

set(CMAKE_PLATFORM_HAS_INSTALLNAME 1)
set(CMAKE_SHARED_LINKER_FLAGS "-rpath @executable_path/Frameworks -rpath @loader_path/Frameworks")
set(CMAKE_SHARED_LIBRARY_CREATE_C_FLAGS "-dynamiclib -Wl,-headerpad_max_install_names")
set(CMAKE_SHARED_MODULE_CREATE_C_FLAGS "-bundle -Wl,-headerpad_max_install_names")
set(CMAKE_SHARED_MODULE_LOADER_C_FLAG "-Wl,-bundle_loader,")
set(CMAKE_SHARED_MODULE_LOADER_CXX_FLAG "-Wl,-bundle_loader,")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".tbd" ".dylib" ".so" ".a")
set(CMAKE_SHARED_LIBRARY_SONAME_C_FLAG "-install_name")

set(CMAKE_FIND_ROOT_PATH
        ${CMAKE_OSX_SYSROOT_INT} ${CMAKE_PREFIX_PATH}
        CACHE STRING "Root path that will be prepended to all search paths")

set(CMAKE_FIND_FRAMEWORK FIRST)

set(CMAKE_FRAMEWORK_PATH
        ${CMAKE_DEVELOPER_ROOT}/Library/PrivateFrameworks
        ${CMAKE_OSX_SYSROOT_INT}/System/Library/Frameworks
        ${CMAKE_FRAMEWORK_PATH} CACHE STRING "Frameworks search paths" ${FORCE_CACHE})

if(NOT CMAKE_FIND_ROOT_PATH_MODE_PROGRAM)
    set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM BOTH CACHE STRING "" ${FORCE_CACHE})
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_LIBRARY)
    set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY CACHE STRING "" ${FORCE_CACHE})
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_INCLUDE)
    set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY CACHE STRING "" ${FORCE_CACHE})
endif()
if(NOT CMAKE_FIND_ROOT_PATH_MODE_PACKAGE)
    set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY CACHE STRING "" ${FORCE_CACHE})
endif()
