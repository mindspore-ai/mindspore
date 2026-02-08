if(MSVC)
    set(opencv_CXXFLAGS "/DWIN32 /D_WINDOWS /W3 /GR /EHsc /std:c++17")
    set(opencv_CFLAGS "${CMAKE_C_FLAGS}")
    set(opencv_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
    if(DEBUG_MODE)
        set(opencv_Debug ON)
    endif()
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(opencv_CXXFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_LDFLAGS "-Wl")
elseif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
    set(opencv_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_CXXFLAGS "${opencv_CXXFLAGS} -Wno-attributes -Wno-unknown-pragmas")
    set(opencv_CXXFLAGS "${opencv_CXXFLAGS} -Wno-unused-value -Wno-implicit-fallthrough")
else()
    set(opencv_CXXFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2")
    set(opencv_CXXFLAGS "${opencv_CXXFLAGS} -O2")
    if(NOT ENABLE_GLIBCXX)
        set(opencv_CXXFLAGS "${opencv_CXXFLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
    endif()
    set(opencv_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -D_FORTIFY_SOURCE=2 -O2")
    set(opencv_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack -s")
endif()

if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/opencv.git")
    set(GIT_TAG "openEuler-22.03-LTS")
    set(SHA256 "d8b00a5440c8e5d275aa5b141f89d69ee196d9dcd2d2032ddd8ef4b04010999a")
    set(OPENCV_SRC "${CMAKE_BINARY_DIR}/_deps/opencv-src")
    __download_pkg_with_git(opencv ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    execute_process(COMMAND tar -xf ${OPENCV_SRC}/opencv-4.5.2.tar.gz --strip-components 1 -C ${OPENCV_SRC})
else()
    if(ENABLE_GITEE)
        set(REQ_URL "https://gitee.com/mirrors/opencv/repository/archive/4.11.0.tar.gz")
        set(SHA256 "9a7c11f924eff5f8d8070e297b322ee68b9227e003fd600d4b8122198091665f")
    else()
        set(REQ_URL "https://github.com/opencv/opencv/archive/refs/tags/4.11.0.tar.gz")
        set(SHA256 "9a7c11f924eff5f8d8070e297b322ee68b9227e003fd600d4b8122198091665f")
    endif()
endif()

if(MSVC)
    mindspore_add_pkg(opencv
            VER 4.11.0
            LIBS opencv_core4110 opencv_imgcodecs4110 opencv_imgproc4110
            LIB_PATH x64/*/lib
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
            -DWITH_ADE=OFF
            -DBUILD_ZLIB=ON
            -DBUILD_JPEG=ON
            -DBUILD_PNG=ON
            -DWITH_OPENEXR=OFF
            -DBUILD_TESTS=OFF
            -DBUILD_PERF_TESTS=OFF
            -DBUILD_opencv_apps=OFF
            -DCMAKE_SKIP_RPATH=TRUE
            -DBUILD_opencv_python3=OFF
            -DBUILD_opencv_videoio=OFF
            -DWITH_FFMPEG=OFF
            -DWITH_TIFF=ON
            -DBUILD_TIFF=ON
            -DWITH_JASPER=OFF
            -DBUILD_JASPER=OFF
            -DCV_TRACE=OFF    # cause memory usage increacing
            -DWITH_OPENJPEG=OFF
            PATCHES ${TOP_DIR}/third_party/patch/opencv/libtiff/CVE-2024-7006.patch
            PATCHES ${TOP_DIR}/third_party/patch/opencv/Fix_Binary.patch)
elseif(WIN32)
    mindspore_add_pkg(opencv
                VER 4.11.0
                LIBS libopencv_core4110.dll.a libopencv_imgcodecs4110.dll.a libopencv_imgproc4110.dll.a
                LIB_PATH x64/mingw/lib
                URL ${REQ_URL}
                SHA256 ${SHA256}
                CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
                -DWITH_ADE=OFF
                -DBUILD_ZLIB=ON
                -DBUILD_JPEG=ON
                -DBUILD_PNG=ON
                -DWITH_OPENEXR=OFF
                -DBUILD_TESTS=OFF
                -DBUILD_PERF_TESTS=OFF
                -DBUILD_opencv_apps=OFF
                -DCMAKE_SKIP_RPATH=TRUE
                -DBUILD_opencv_python3=OFF
                -DBUILD_opencv_videoio=OFF
                -DWITH_FFMPEG=OFF
                -DWITH_TIFF=ON
                -DBUILD_TIFF=ON
                -DWITH_JASPER=OFF
                -DBUILD_JASPER=OFF
                -DCV_TRACE=OFF    # cause memory usage increacing
                -DWITH_LAPACK=OFF
                -DWITH_OPENJPEG=OFF
                PATCHES ${TOP_DIR}/third_party/patch/opencv/libtiff/CVE-2024-7006.patch
                PATCHES ${TOP_DIR}/third_party/patch/opencv/Fix_Binary.patch)
elseif(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    mindspore_add_pkg(opencv
            VER 4.11.0
            LIBS opencv_core opencv_imgcodecs opencv_imgproc
            URL ${REQ_URL}
            SHA256  ${SHA256}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
            -DWITH_ADE=OFF
            -DBUILD_ZLIB=ON
            -DBUILD_JPEG=ON
            -DBUILD_PNG=ON
            -DWITH_OPENEXR=OFF
            -DBUILD_TESTS=OFF
            -DBUILD_PERF_TESTS=OFF
            -DBUILD_opencv_apps=OFF
            -DCMAKE_SKIP_RPATH=TRUE
            -DBUILD_opencv_python3=OFF
            -DWITH_FFMPEG=OFF
            -DWITH_TIFF=ON
            -DBUILD_TIFF=ON
            -DWITH_JASPER=OFF
            -DBUILD_JASPER=OFF
            -DCV_TRACE=OFF    # cause memory usage increacing
            -DWITH_LAPACK=OFF
            -DWITH_OPENJPEG=OFF
            -DCPU_BASELINE=NEON,FP16,NEON_FP16   # The environment does not support the "NEON_DOTPROD" instruction set.
            -DCPU_DISPATCH=                      # The environment does not support the "NEON_DOTPROD" instruction set.
            PATCHES ${TOP_DIR}/third_party/patch/opencv/libtiff/CVE-2024-7006.patch
            PATCHES ${TOP_DIR}/third_party/patch/opencv/Fix_Binary.patch)
else()
    mindspore_add_pkg(opencv
                VER 4.11.0
                LIBS opencv_core opencv_imgcodecs opencv_imgproc
                URL ${REQ_URL}
                SHA256  ${SHA256}
                CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
                -DWITH_ADE=OFF
                -DBUILD_ZLIB=ON
                -DBUILD_JPEG=ON
                -DBUILD_PNG=ON
                -DWITH_OPENEXR=OFF
                -DBUILD_TESTS=OFF
                -DBUILD_PERF_TESTS=OFF
                -DBUILD_opencv_apps=OFF
                -DCMAKE_SKIP_RPATH=TRUE
                -DBUILD_opencv_python3=OFF
                -DWITH_FFMPEG=OFF
                -DWITH_TIFF=ON
                -DBUILD_TIFF=ON
                -DWITH_JASPER=OFF
                -DBUILD_JASPER=OFF
                -DCV_TRACE=OFF    # cause memory usage increacing
                -DWITH_LAPACK=OFF
                -DWITH_OPENJPEG=OFF
                PATCHES ${TOP_DIR}/third_party/patch/opencv/libtiff/CVE-2024-7006.patch
                PATCHES ${TOP_DIR}/third_party/patch/opencv/Fix_Binary.patch)
endif()

if(MSVC)
    include_directories(${opencv_INC})
    add_library(mindspore::opencv_core ALIAS opencv::opencv_core4110)
    add_library(mindspore::opencv_imgcodecs ALIAS opencv::opencv_imgcodecs4110)
    add_library(mindspore::opencv_imgproc ALIAS opencv::opencv_imgproc4110)
elseif(WIN32)
    include_directories(${opencv_INC})
    add_library(mindspore::opencv_core ALIAS opencv::libopencv_core4110.dll.a)
    add_library(mindspore::opencv_imgcodecs ALIAS opencv::libopencv_imgcodecs4110.dll.a)
    add_library(mindspore::opencv_imgproc ALIAS opencv::libopencv_imgproc4110.dll.a)
else()
    include_directories(${opencv_INC}/opencv4)
    add_library(mindspore::opencv_core ALIAS opencv::opencv_core)
    add_library(mindspore::opencv_imgcodecs ALIAS opencv::opencv_imgcodecs)
    add_library(mindspore::opencv_imgproc ALIAS opencv::opencv_imgproc)
endif()
