set(FFMPEG_FLAGS
        --disable-programs
        --disable-doc
        --disable-debug
        --disable-avdevice
        --disable-postproc
        --disable-avfilter
        --disable-network
        --disable-encoders
        --disable-hwaccels
        --disable-muxers
        --disable-bsfs
        --disable-protocols
        --enable-protocol=file
        --enable-protocol=pipe
        --disable-indevs
        --disable-outdevs
        --disable-devices
        --disable-filters
        --disable-bzlib
        --disable-iconv
        --disable-libxcb
        --disable-lzma
        --disable-sdl2
        --disable-xlib
        --disable-zlib)

set(REQ_URL "https://github.com/FFmpeg/FFmpeg/archive/n4.3.1.tar.gz")
set(MD5 "426ca412ca61634a248c787e29507206")

mindspore_add_pkg(ffmpeg
        VER 4.3.1
        LIBS avcodec avformat avutil swresample swscale
        URL ${REQ_URL}
        MD5 ${MD5}
        CONFIGURE_COMMAND ./configure --disable-static --enable-shared --disable-x86asm ${FFMPEG_FLAGS}
        )

include_directories(${ffmpeg_INC})
add_library(mindspore::avcodec ALIAS ffmpeg::avcodec)
add_library(mindspore::avformat ALIAS ffmpeg::avformat)
add_library(mindspore::avutil ALIAS ffmpeg::avutil)
add_library(mindspore::swresample ALIAS ffmpeg::swresample)
add_library(mindspore::swscale ALIAS ffmpeg::swscale)
