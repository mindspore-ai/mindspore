set(jpeg_turbo_USE_STATIC_LIBS ON)
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(jpeg_turbo_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 -O2")
else()
    set(jpeg_turbo_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 -O2")
endif()

set(jpeg_turbo_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
mindspore_add_pkg(jpeg_turbo
        VER 2.0.4
        LIBS jpeg
        URL https://github.com/libjpeg-turbo/libjpeg-turbo/archive/2.0.4.tar.gz
        MD5 44c43e4a9fb352f47090804529317c88
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DCMAKE_SKIP_RPATH=TRUE
        )
include_directories(${jpeg_turbo_INC})
add_library(mindspore::jpeg_turbo ALIAS jpeg_turbo::jpeg)
