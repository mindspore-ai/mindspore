if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/abseil-cpp/repository/archive/20200923.3.tar.gz")
    set(MD5 "425588c377891eaf368cfad0ea934601")
else()
    set(REQ_URL "https://github.com/abseil/abseil-cpp/archive/20200923.3.tar.gz")
    set(MD5 "daba6e99c7a84e2242a0107bbd873669")
endif()

if(NOT ENABLE_GLIBCXX)
    set(absl_CXXFLAGS "${absl_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
endif()

mindspore_add_pkg(absl
        VER 20200923.3
        LIBS absl_strings absl_throw_delegate absl_raw_logging_internal absl_int128 absl_bad_optional_access
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=TRUE)

include_directories(${absl_INC})

add_library(mindspore::absl_strings ALIAS absl::absl_strings)
add_library(mindspore::absl_throw_delegate ALIAS absl::absl_throw_delegate)
add_library(mindspore::absl_raw_logging_internal ALIAS absl::absl_raw_logging_internal)
add_library(mindspore::absl_int128 ALIAS absl::absl_int128)
add_library(mindspore::absl_bad_optional_access ALIAS absl::absl_bad_optional_access)
