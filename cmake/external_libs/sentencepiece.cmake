if (WIN32)
        set(sentencepiece_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result -Wno-stringop-overflow -Wno-format-extra-args -Wno-format")
        set(sentencepiece_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
        mindspore_add_pkg(sentencepiece
                VER 0.1.92
                LIBS sentencepiece sentencepiece_train
                URL https://github.com/google/sentencepiece/archive/v0.1.92.tar.gz
                CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=ON
                MD5 5dfd2241914b5598a68b2a8542ed8e91
                )
else ()
       set(sentencepiece_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2 -Wno-unused-result -Wno-sign-compare")
       set(sentencepiece_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
       mindspore_add_pkg(sentencepiece
                VER 0.1.92
                LIBS sentencepiece sentencepiece_train
                URL https://github.com/google/sentencepiece/archive/v0.1.92.tar.gz
                CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DSPM_USE_BUILTIN_PROTOBUF=OFF -DSPM_ENABLE_SHARED=OFF  -DPROTOBUF_INC=${protobuf_INC}
                MD5 5dfd2241914b5598a68b2a8542ed8e91
                PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sentencepiece/sentencepiece.patch001
                )
endif ()
include_directories(${sentencepiece_INC})
add_library(mindspore::sentencepiece ALIAS sentencepiece::sentencepiece)
add_library(mindspore::sentencepiece_train ALIAS sentencepiece::sentencepiece_train)