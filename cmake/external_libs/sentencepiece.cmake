set(ENABLE_GITEE_EULER OFF)
if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/sentencepiece.git")
    set(GIT_TAG "master")
    set(MD5 "4f88df28544b5f1a351f3dbf6b6413b8")
    set(SENTENCEPIECE_SRC "${TOP_DIR}/build/mindspore/_deps/sentencepiece-src")
    __download_pkg_with_git(sentencepiece ${GIT_REPOSITORY} ${GIT_TAG} ${MD5})
    execute_process(COMMAND tar -xf ${SENTENCEPIECE_SRC}/v0.1.92.tar.gz --strip-components 1 -C ${SENTENCEPIECE_SRC})
else()
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/sentencepiece/repository/archive/v0.1.92.tar.gz")
    set(MD5 "0fc99de9f09b9184398f49647791799f")
else()
    set(REQ_URL "https://github.com/google/sentencepiece/archive/v0.1.92.tar.gz")
    set(MD5 "5dfd2241914b5598a68b2a8542ed8e91")
endif()
endif()


if(WIN32)
    set(sentencepiece_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result -Wno-stringop-overflow \
        -Wno-format-extra-args -Wno-format")
    set(sentencepiece_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
    mindspore_add_pkg(sentencepiece
        VER 0.1.92
        LIBS sentencepiece sentencepiece_train
        URL ${REQ_URL}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=ON -DSPM_ENABLE_SHARED=OFF
        MD5 ${MD5}
        )
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
            MD5 ${MD5}
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sentencepiece/sentencepiece.patch001_cpu
            )
    else()
        mindspore_add_pkg(sentencepiece
            VER 0.1.92
            LIBS sentencepiece sentencepiece_train
            URL ${REQ_URL}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=OFF -DSPM_ENABLE_SHARED=OFF
                -DPROTOBUF_INC=${protobuf_INC}
            MD5 ${MD5}
            PATCHES ${TOP_DIR}/third_party/patch/sentencepiece/sentencepiece.patch001
            )
    endif()
endif()
include_directories(${sentencepiece_INC})
add_library(mindspore::sentencepiece ALIAS sentencepiece::sentencepiece)
add_library(mindspore::sentencepiece_train ALIAS sentencepiece::sentencepiece_train)

