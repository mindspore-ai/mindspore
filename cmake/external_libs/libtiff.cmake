if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(tiff_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -Wno-unused-result \
    -Wno-unused-but-set-variable -fPIC -D_FORTIFY_SOURCE=2 -O2")
    set(tiff_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -Wno-unused-result \
    -Wno-unused-but-set-variable -fPIC -D_FORTIFY_SOURCE=2 -O2")
else()
    set(tiff_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -Wno-unused-result \
        -Wno-unused-but-set-variable -fPIC -D_FORTIFY_SOURCE=2 -O2")
    set(tiff_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -Wno-unused-result \
        -Wno-unused-but-set-variable -fPIC -D_FORTIFY_SOURCE=2 -O2")
    if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(tiff_CFLAGS "${tiff_CFLAGS} -Wno-int-to-pointer-cast -Wno-implicit-fallthrough -Wno-pointer-to-int-cast")
    endif()
endif()

set(tiff_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/libtiff/repository/archive/v4.1.0.tar.gz")
    set(MD5 "76b951159c8bdf47dba4803659c232d1")
else()
    set(REQ_URL "https://gitlab.com/libtiff/libtiff/-/archive/v4.1.0/libtiff-v4.1.0.tar.gz")
    set(MD5 "21de8d35c1b21ac82663fa9f56d3350d")
endif()

mindspore_add_pkg(tiff
        VER 4.1.0
        LIBS tiff
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -Djbig=OFF -Dlzma=OFF -Djpeg12=OFF -Dzstd=OFF -Dpixarlog=OFF
        -Dold-jpeg=OFF -Dwebp=OFF -DBUILD_SHARED_LIBS=OFF)
message("tiff include = ${tiff_INC}")
message("tiff lib = ${tiff_LIB}")
