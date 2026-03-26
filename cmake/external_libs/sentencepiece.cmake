if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/sentencepiece.git")
    set(GIT_TAG "master")
    set(SHA256 "4f88df28544b5f1a351f3dbf6b6413b8")
    set(SENTENCEPIECE_SRC "${TOP_DIR}/build/mindspore/_deps/sentencepiece-src")
    __download_pkg_with_git(sentencepiece ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    execute_process(COMMAND tar -xf ${SENTENCEPIECE_SRC}/v0.1.98.tar.gz --strip-components 1 -C ${SENTENCEPIECE_SRC})
else()
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/sentencepiece/repository/archive/v0.1.98.tar.gz")
    set(SHA256 "623cb16369626c14ddebd02323dc35b87c63717fbf5c1c684f999a9f0c1e3c6a")
else()
    set(REQ_URL "https://github.com/google/sentencepiece/archive/v0.1.98.tar.gz")
    set(SHA256 "e8e09beffacd9667ed40c4652306f7e7990100164dfa26d8bd8a66b097471cb2")
endif()
endif()

if(EXISTS ${absl_ROOT}/lib64)
    set(_FINDPACKAGE_ABSL_CONFIG_DIR_SPM "${absl_ROOT}/lib64/cmake/absl")
else()
    set(_FINDPACKAGE_ABSL_CONFIG_DIR_SPM "${absl_ROOT}/lib/cmake/absl")
endif()
if(EXISTS ${protobuf_ROOT}/lib64)
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR_SPM "${protobuf_ROOT}/lib64/cmake/protobuf")
    set(_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR_SPM "${protobuf_ROOT}/lib64/cmake/utf8_range")
else()
    set(_FINDPACKAGE_PROTOBUF_CONFIG_DIR_SPM "${protobuf_ROOT}/lib/cmake/protobuf")
    set(_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR_SPM "${protobuf_ROOT}/lib/cmake/utf8_range")
endif()


if(WIN32)
    if(MSVC)
        set(sentencepiece_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 /EHsc")
    else()
        set(sentencepiece_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result -Wno-stringop-overflow \
            -Wno-format-extra-args -Wno-format")
    endif()

    set(sentencepiece_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    if(MSVC)
        mindspore_add_pkg(sentencepiece
            VER 0.1.98
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=OFF -DSPM_ENABLE_SHARED=OFF
                -DSPM_USE_EXTERNAL_ABSL=ON
                -DPROTOBUF_INC=${protobuf_INC}
                -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR_SPM}
                -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR_SPM}
                -Dutf8_range_DIR:PATH=${_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR_SPM}
                -DABSL_INC=${absl_INC}
            SHA256 ${SHA256}
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sentencepiece/sentencepiece_msvc.patch001
            )
    else()
        mindspore_add_pkg(sentencepiece
            VER 0.1.98
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=ON -DSPM_ENABLE_SHARED=OFF
            SHA256 ${SHA256}
            )
    endif()
else()
    set(sentencepiece_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result -Wno-sign-compare -I${absl_INC}")
    set(sentencepiece_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    if(ENABLE_GLIBCXX)
        mindspore_add_pkg(sentencepiece
            VER 0.1.98
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=OFF -DSPM_ENABLE_SHARED=OFF
                -DSPM_USE_EXTERNAL_ABSL=ON
                -DPROTOBUF_INC=${protobuf_INC} -DCMAKE_CXX_STANDARD=17
                -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR_SPM}
                -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR_SPM}
                -Dutf8_range_DIR:PATH=${_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR_SPM}
                -DABSL_INC=${absl_INC}
            SHA256 ${SHA256}
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sentencepiece/sentencepiece.patch001_cpu
            )
    else()
        mindspore_add_pkg(sentencepiece
            VER 0.1.98
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=OFF -DSPM_ENABLE_SHARED=OFF
                -DSPM_USE_EXTERNAL_ABSL=ON
                -DPROTOBUF_INC=${protobuf_INC}
                -DCMAKE_CXX_STANDARD=17
                -Dabsl_DIR:PATH=${_FINDPACKAGE_ABSL_CONFIG_DIR_SPM}
                -DProtobuf_DIR:PATH=${_FINDPACKAGE_PROTOBUF_CONFIG_DIR_SPM}
                -Dutf8_range_DIR:PATH=${_FINDPACKAGE_UTF8_RANGE_CONFIG_DIR_SPM}
                -DABSL_INC=${absl_INC}
            SHA256 ${SHA256}
            PATCHES ${TOP_DIR}/third_party/patch/sentencepiece/sentencepiece.patch001
            )
    endif()
endif()
include_directories(${sentencepiece_INC})
add_library(mindspore::sentencepiece ALIAS sentencepiece::sentencepiece)
add_library(mindspore::sentencepiece_train ALIAS sentencepiece::sentencepiece_train)

