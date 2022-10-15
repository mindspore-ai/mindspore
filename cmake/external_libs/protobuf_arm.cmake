set(protobuf_arm_USE_STATIC_LIBS ON)
if(BUILD_LITE)
    if(MSVC)
        set(protobuf_arm_CXXFLAGS "${CMAKE_CXX_FLAGS}")
        set(protobuf_arm_CFLAGS "${CMAKE_C_FLAGS}")
        set(protobuf_arm_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
        set(_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX ${CMAKE_STATIC_LIBRARY_PREFIX})
        set(CMAKE_STATIC_LIBRARY_PREFIX "lib")
    else()
        set(protobuf_arm_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_arm_CXXFLAGS "${protobuf_arm_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
        set(protobuf_arm_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
    endif()
else()
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(protobuf_arm_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC \
            -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
        set(protobuf_arm_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
    else()
        set(protobuf_arm_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter \
            -fPIC -fvisibility=hidden -D_FORTIFY_SOURCE=2 -O2")
        if(NOT ENABLE_GLIBCXX)
            set(protobuf_arm_CXXFLAGS "${protobuf_arm_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        endif()
    endif()
    set(protobuf_arm_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

set(_ms_tmp_CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS ${_ms_tmp_CMAKE_CXX_FLAGS})
string(REPLACE " -Wall" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
string(REPLACE " -Werror" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/protobuf_source/repository/archive/v3.13.0.tar.gz")
    set(MD5 "53ab10736257b3c61749de9800b8ce97")
else()
    set(REQ_URL "https://github.com/protocolbuffers/protobuf/archive/v3.13.0.tar.gz")
    set(MD5 "1a6274bc4a65b55a6fa70e264d796490")
endif()

if(BUILD_LITE)
    set(PROTOBUF_PATCH_ROOT ${TOP_DIR}/third_party/patch/protobuf)
else()
    set(PROTOBUF_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/protobuf)
endif()

if(APPLE)
    mindspore_add_pkg(protobuf_arm
            VER 3.13.0
            LIBS protobuf
            URL ${REQ_URL}
            MD5 ${MD5}
            CMAKE_PATH cmake/
            CMAKE_OPTION
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_SHARED_LIBS=OFF
            -DCMAKE_BUILD_TYPE=Release
            -Dprotobuf_WITH_ZLIB=OFF
            -DCMAKE_OSX_SYSROOT=${CMAKE_OSX_SYSROOT}
            -DCMAKE_OSX_DEPLOYMENT_TARGET=${CMAKE_OSX_DEPLOYMENT_TARGET}
            PATCHES ${PROTOBUF_PATCH_ROOT}/CVE-2021-22570.patch
            PATCHES ${PROTOBUF_PATCH_ROOT}/CVE-2022-1941.patch)
else()
    mindspore_add_pkg(protobuf_arm
            VER 3.13.0
            LIBS protobuf
            URL ${REQ_URL}
            MD5 ${MD5}
            CMAKE_PATH cmake/
            CMAKE_OPTION
            -Dprotobuf_BUILD_TESTS=OFF
            -Dprotobuf_BUILD_SHARED_LIBS=OFF
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
            -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
            -Dprotobuf_WITH_ZLIB=OFF
            PATCHES ${PROTOBUF_PATCH_ROOT}/CVE-2021-22570.patch
            PATCHES ${PROTOBUF_PATCH_ROOT}/CVE-2022-1941.patch)
endif()

include_directories(${protobuf_arm_INC})
add_library(mindspore::protobuf_arm ALIAS protobuf_arm::protobuf)
set(CMAKE_CXX_FLAGS  ${_ms_tmp_CMAKE_CXX_FLAGS})
if(MSVC)
    set(CMAKE_STATIC_LIBRARY_PREFIX, ${_ms_tmp_CMAKE_STATIC_LIBRARY_PREFIX})
endif()
