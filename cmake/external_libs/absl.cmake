if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/abseil-cpp.git")
    set(GIT_TAG "openEuler-22.03-LTS")
    set(SHA256 "365b1ecbbcd81b4c58101808a8a28a3cf9ad7f9d05c08080a35c0d4283a44afa")
    set(ABSL_SRC "${TOP_DIR}/build/mindspore/_deps/absl-src")
    __download_pkg_with_git(absl ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    execute_process(COMMAND tar -xf ${ABSL_SRC}/abseil-cpp-20210324.2.tar.gz --strip-components 1 -C ${ABSL_SRC})
else()
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/abseil-cpp/repository/archive/20250127.1.tar.gz")
    set(SHA256 "b396401fd29e2e679cace77867481d388c807671dc2acc602a0259eeb79b7811")
else()
    set(REQ_URL "https://github.com/abseil/abseil-cpp/archive/20250127.1.tar.gz")
    set(SHA256 "b396401fd29e2e679cace77867481d388c807671dc2acc602a0259eeb79b7811")
endif()
endif()

if(NOT ENABLE_GLIBCXX)
    set(absl_CXXFLAGS "${absl_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

mindspore_add_pkg(absl
        VER 20250127.1
        LIBS absl_strings absl_throw_delegate absl_raw_logging_internal absl_int128 absl_bad_optional_access
        absl_log_internal_check_op
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE
        -DCMAKE_CXX_STANDARD=17
        -DCMAKE_CXX_STANDARD_REQUIRED=ON
        -DCMAKE_CXX_EXTENSIONS=OFF
        -DCMAKE_INSTALL_LIBDIR=lib
        )

include_directories(${absl_INC})

add_library(mindspore::absl_strings ALIAS absl::absl_strings)
add_library(mindspore::absl_throw_delegate ALIAS absl::absl_throw_delegate)
add_library(mindspore::absl_raw_logging_internal ALIAS absl::absl_raw_logging_internal)
add_library(mindspore::absl_int128 ALIAS absl::absl_int128)
add_library(mindspore::absl_bad_optional_access ALIAS absl::absl_bad_optional_access)
add_library(mindspore::absl_log_internal_check_op ALIAS absl::absl_log_internal_check_op)
