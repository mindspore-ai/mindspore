if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(opencv_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_LDFLAGS "-Wl")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(opencv_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_CXXFLAGS "${opencv_CXXFLAGS} -Wno-attributes -Wno-unknown-pragmas")
    set(opencv_CXXFLAGS "${opencv_CXXFLAGS} -Wno-unused-value -Wno-implicit-fallthrough")
else()
    set(opencv_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=0 -O2")
    set(opencv_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
endif()

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/opencv/repository/archive/4.2.0.tar.gz")
    set(MD5 "00424c7c4acde1e26ebf17aaa155bf23")
else()
    set(REQ_URL "https://github.com/opencv/opencv/archive/4.2.0.tar.gz")
    set(MD5 "e8cb208ce2723481408b604b480183b6")
endif()

if(WIN32)
    mindspore_add_pkg(opencv
            VER 4.2.0
            LIBS libopencv_core420.dll.a libopencv_imgcodecs420.dll.a libopencv_imgproc420.dll.a
            LIB_PATH x64/mingw/lib
            URL ${REQ_URL}
            MD5 ${MD5}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF -DWITH_ADE=OFF
            -DBUILD_ZLIB=ON
            -DBUILD_JPEG=ON
            -DBUILD_PNG=ON
            -DBUILD_OPENEXR=ON
            -DBUILD_TESTS=OFF
            -DBUILD_PERF_TESTS=OFF
            -DBUILD_opencv_apps=OFF
            -DCMAKE_SKIP_RPATH=TRUE
            -DBUILD_opencv_python3=OFF
            -DBUILD_opencv_videoio=OFF
            -DWITH_FFMPEG=OFF
            -DWITH_TIFF=ON
            -DBUILD_TIFF=OFF
            -DWITH_JASPER=OFF
            -DBUILD_JASPER=OFF
            -DTIFF_INCLUDE_DIR=${tiff_INC}
            -DTIFF_LIBRARY=${tiff_LIB})
else()
    mindspore_add_pkg(opencv
            VER 4.2.0
            LIBS opencv_core opencv_imgcodecs opencv_imgproc
            URL ${REQ_URL}
            MD5  ${MD5}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF -DWITH_ADE=OFF
            -DBUILD_ZLIB=ON
            -DBUILD_JPEG=ON
            -DBUILD_PNG=ON
            -DBUILD_OPENEXR=ON
            -DBUILD_TESTS=OFF
            -DBUILD_PERF_TESTS=OFF
            -DBUILD_opencv_apps=OFF
            -DCMAKE_SKIP_RPATH=TRUE
            -DBUILD_opencv_python3=OFF
            -DWITH_FFMPEG=OFF
            -DWITH_TIFF=ON
            -DBUILD_TIFF=OFF
            -DWITH_JASPER=OFF
            -DBUILD_JASPER=OFF
            -DTIFF_INCLUDE_DIR=${tiff_INC}
            -DTIFF_LIBRARY=${tiff_LIB})
endif()

if(WIN32)
    include_directories(${opencv_INC})
    add_library(mindspore::opencv_core ALIAS opencv::libopencv_core420.dll.a)
    add_library(mindspore::opencv_imgcodecs ALIAS opencv::libopencv_imgcodecs420.dll.a)
    add_library(mindspore::opencv_imgproc ALIAS opencv::libopencv_imgproc420.dll.a)
else()
    include_directories(${opencv_INC}/opencv4)
    add_library(mindspore::opencv_core ALIAS opencv::opencv_core)
    add_library(mindspore::opencv_imgcodecs ALIAS opencv::opencv_imgcodecs)
    add_library(mindspore::opencv_imgproc ALIAS opencv::opencv_imgproc)
endif()
