
set(tiff_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -Wno-unused-result \
    -Wno-unused-but-set-variable -fPIC -D_FORTIFY_SOURCE=2 -O2")
set(tiff_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -Wno-unused-result \
    -Wno-unused-but-set-variable -fPIC -D_FORTIFY_SOURCE=2 -O2")
set(tiff_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")

mindspore_add_pkg(tiff
        VER 4.1.0
        LIBS tiff
        URL https://gitlab.com/libtiff/libtiff/-/archive/v4.1.0/libtiff-v4.1.0.tar.gz
        MD5 21de8d35c1b21ac82663fa9f56d3350d
        CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -Djbig=OFF -Dlzma=OFF -Djpeg12=OFF -Dzstd=OFF -Dpixarlog=OFF
        -Dold-jpeg=OFF -Dwebp=OFF -DBUILD_SHARED_LIBS=OFF)
message("tiff include = ${tiff_INC}")
message("tiff lib = ${tiff_LIB}")
