if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    set(FFMPEG_FLAGS
        --disable-programs
        --disable-doc
        --disable-postproc
        --disable-decoder=av1
        --disable-libxcb
        --disable-hwaccels
        --disable-static
        --enable-shared
        --disable-x86asm
        --extra-cflags="-fstack-protector-all"
        --extra-cxxflags="-fstack-protector-all"
        --extra-cflags="-D_FORTIFY_SOURCE=2"
        --extra-cxxflags="-D_FORTIFY_SOURCE=2"
        --extra-ldflags="-Wl,-z,relro,-z,now")
else()
    set(FFMPEG_FLAGS
        --disable-programs
        --disable-doc
        --disable-postproc
        --disable-decoder=av1
        --disable-libxcb
        --disable-hwaccels
        --disable-static
        --enable-shared
        --disable-x86asm)
endif()

set(REQ_URL "https://ffmpeg.org/releases/ffmpeg-5.1.4.tar.gz")
set(SHA256 "ad1d83e0f022ba405928ea4985c9cab978c4698bb7937738a1085808c3eb78b4")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    set(FFMPEG_DLL_DIR $ENV{FFMPEG_CACHE_DIR})
    add_library(mindspore::avcodec SHARED IMPORTED GLOBAL)
    add_library(mindspore::avdevice SHARED IMPORTED GLOBAL)
    add_library(mindspore::avfilter SHARED IMPORTED GLOBAL)
    add_library(mindspore::avformat SHARED IMPORTED GLOBAL)
    add_library(mindspore::avutil SHARED IMPORTED GLOBAL)
    add_library(mindspore::swresample SHARED IMPORTED GLOBAL)
    add_library(mindspore::swscale SHARED IMPORTED GLOBAL)
    set_target_properties(mindspore::avcodec PROPERTIES IMPORTED_IMPLIB_RELEASE ${FFMPEG_DLL_DIR}/bin/avcodec.lib)
    set_target_properties(mindspore::avdevice PROPERTIES IMPORTED_IMPLIB_RELEASE ${FFMPEG_DLL_DIR}/bin/avdevice.lib)
    set_target_properties(mindspore::avfilter PROPERTIES IMPORTED_IMPLIB_RELEASE ${FFMPEG_DLL_DIR}/bin/avfilter.lib)
    set_target_properties(mindspore::avformat PROPERTIES IMPORTED_IMPLIB_RELEASE ${FFMPEG_DLL_DIR}/bin/avformat.lib)
    set_target_properties(mindspore::avutil PROPERTIES IMPORTED_IMPLIB_RELEASE ${FFMPEG_DLL_DIR}/bin/avutil.lib)
    set_target_properties(mindspore::swresample PROPERTIES IMPORTED_IMPLIB_RELEASE ${FFMPEG_DLL_DIR}/bin/swresample.lib)
    set_target_properties(mindspore::swscale PROPERTIES IMPORTED_IMPLIB_RELEASE ${FFMPEG_DLL_DIR}/bin/swscale.lib)
    include_directories(${FFMPEG_DLL_DIR}/include)
else()
    mindspore_add_pkg(ffmpeg
            VER 5.1.4
            LIBS avcodec avdevice avfilter avformat avutil swresample swscale
            URL ${REQ_URL}
            SHA256 ${SHA256}
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-7272.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2023-50008.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2023-51795.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2023-51793.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-35366.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-35367.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-35368.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-31578.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-36616.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-36617.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-36618.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-36613.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2025-22919.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2025-22921.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2025-25473.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2025-1594.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2024-7055.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2025-59733.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2025-59732.patch
            PATCHES ${TOP_DIR}/third_party/patch/ffmpeg/CVE-2025-59731.patch
            CONFIGURE_COMMAND ./configure ${FFMPEG_FLAGS}
            )

    include_directories(${ffmpeg_INC})

    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        add_custom_target(
                link_ffmpeg ALL
                COMMAND echo "modify ffmpeg install name"
                COMMENT "modify ffmpeg install name"
        )

        # rename self
        function(change_ffmpeg_dylib_name lib_name lib)
            add_custom_command(
                TARGET link_ffmpeg
                POST_BUILD
                COMMAND install_name_tool -id @rpath/lib${lib_name}.dylib $<TARGET_FILE:ffmpeg::${lib}>
                VERBATIM
            )
        endfunction()

        change_ffmpeg_dylib_name("avutil.57" "avutil")
        change_ffmpeg_dylib_name("swresample.4" "swresample")
        change_ffmpeg_dylib_name("swscale.6" "swscale")
        change_ffmpeg_dylib_name("avformat.59" "avformat")
        change_ffmpeg_dylib_name("avfilter.8" "avfilter")
        change_ffmpeg_dylib_name("avdevice.59" "avdevice")
        change_ffmpeg_dylib_name("avcodec.59" "avcodec")

        # depend
        set(FFMPEG_PATH ${ffmpeg_LIBPATH})

        function(change_ffmpeg_dylib_depend target_dylib link_dylibs)
            foreach(dylib ${link_dylibs})
              set(SRC_NAME ${FFMPEG_PATH}/lib${dylib}.dylib)
              set(DST_NAME @rpath/lib${dylib}.dylib)
              add_custom_command(
                TARGET link_ffmpeg
                POST_BUILD
                COMMAND install_name_tool -change ${SRC_NAME} ${DST_NAME} $<TARGET_FILE:ffmpeg::${target_dylib}>
                VERBATIM
              )
            endforeach()
        endfunction()

        change_ffmpeg_dylib_depend("swresample" "avutil.57")
        change_ffmpeg_dylib_depend("swscale" "avutil.57")
        change_ffmpeg_dylib_depend("avformat" "avutil.57;swresample.4;avcodec.59")
        change_ffmpeg_dylib_depend("avfilter" "avutil.57;swresample.4;avcodec.59;avformat.59;swscale.6")
        change_ffmpeg_dylib_depend("avdevice" "avutil.57;swresample.4;avcodec.59;avformat.59;swscale.6;avfilter.8")
        change_ffmpeg_dylib_depend("avcodec" "avutil.57;swresample.4")
    endif()

    add_library(mindspore::avcodec ALIAS ffmpeg::avcodec)
    add_library(mindspore::avdevice ALIAS ffmpeg::avdevice)
    add_library(mindspore::avfilter ALIAS ffmpeg::avfilter)
    add_library(mindspore::avformat ALIAS ffmpeg::avformat)
    add_library(mindspore::avutil ALIAS ffmpeg::avutil)
    add_library(mindspore::swresample ALIAS ffmpeg::swresample)
    add_library(mindspore::swscale ALIAS ffmpeg::swscale)
endif()
