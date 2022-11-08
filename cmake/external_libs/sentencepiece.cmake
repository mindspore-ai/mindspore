if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/sentencepiece.git")
    set(GIT_TAG "master")
    set(SHA256 "4f88df28544b5f1a351f3dbf6b6413b8")
    set(SENTENCEPIECE_SRC "${TOP_DIR}/build/mindspore/_deps/sentencepiece-src")
    __download_pkg_with_git(sentencepiece ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    execute_process(COMMAND tar -xf ${SENTENCEPIECE_SRC}/v0.1.92.tar.gz --strip-components 1 -C ${SENTENCEPIECE_SRC})
else()
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/sentencepiece/repository/archive/v0.1.92.tar.gz")
    set(SHA256 "650325f998fb97f360bfa886a761fb5cd34d51d684b26ea53edcb5a0d9fa7601")
else()
    set(REQ_URL "https://github.com/google/sentencepiece/archive/v0.1.92.tar.gz")
    set(SHA256 "6e9863851e6277862083518cc9f96211f334215d596fc8c65e074d564baeef0c")
endif()
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
            VER 0.1.92
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=ON -DSPM_ENABLE_SHARED=OFF
            SHA256 ${SHA256}
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sentencepiece/sentencepiece_msvc.patch001
            )
    else()
        mindspore_add_pkg(sentencepiece
            VER 0.1.92
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=ON -DSPM_ENABLE_SHARED=OFF
            SHA256 ${SHA256}
            )
    endif()
else()
    set(sentencepiece_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result -Wno-sign-compare")
    set(sentencepiece_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    if(ENABLE_GLIBCXX)
        mindspore_add_pkg(sentencepiece
            VER 0.1.92
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=OFF -DSPM_ENABLE_SHARED=OFF
                -DPROTOBUF_INC=${protobuf_INC} -DCMAKE_CXX_STANDARD=11
            SHA256 ${SHA256}
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sentencepiece/sentencepiece.patch001_cpu
            )
    else()
        mindspore_add_pkg(sentencepiece
            VER 0.1.92
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=OFF -DSPM_ENABLE_SHARED=OFF
                -DPROTOBUF_INC=${protobuf_INC}
            SHA256 ${SHA256}
            PATCHES ${TOP_DIR}/third_party/patch/sentencepiece/sentencepiece.patch001
            )
    endif()
endif()
include_directories(${sentencepiece_INC})
add_library(mindspore::sentencepiece ALIAS sentencepiece::sentencepiece)
add_library(mindspore::sentencepiece_train ALIAS sentencepiece::sentencepiece_train)

