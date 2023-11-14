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
        )

set(REQ_URL "https://ffmpeg.org/releases/ffmpeg-5.1.2.tar.gz")
set(SHA256 "87fe8defa37ce5f7449e36047171fed5e4c3f4bb73eaccea8c954ee81393581c")

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message("FFmpeg software has been compiled.")
else()
    mindspore_add_pkg(ffmpeg
            VER 5.1.2
            LIBS avcodec avdevice avfilter avformat avutil swresample swscale
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CONFIGURE_COMMAND ./configure ${FFMPEG_FLAGS}
            )

    include_directories(${ffmpeg_INC})
    add_library(mindspore::avcodec ALIAS ffmpeg::avcodec)
    add_library(mindspore::avdevice ALIAS ffmpeg::avdevice)
    add_library(mindspore::avfilter ALIAS ffmpeg::avfilter)
    add_library(mindspore::avformat ALIAS ffmpeg::avformat)
    add_library(mindspore::avutil ALIAS ffmpeg::avutil)
    add_library(mindspore::swresample ALIAS ffmpeg::swresample)
    add_library(mindspore::swscale ALIAS ffmpeg::swscale)
endif()
