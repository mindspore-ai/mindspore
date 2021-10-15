if(MSVC)
    set(opencv_CXXFLAGS "${CMAKE_CXX_FLAGS}")
    set(opencv_CFLAGS "${CMAKE_C_FLAGS}")
    set(opencv_LDFLAGS "${CMAKE_SHARED_LINKER_FLAGS}")
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

if(ENABLE_GITEE)
    if("${BUILD_LITE}" STREQUAL "on" OR PYTHON_VERSION MATCHES "3.9")
        set(REQ_URL "https://gitee.com/mirrors/opencv/repository/archive/4.5.1.tar.gz")
        set(MD5 "59fd2c5449f8a3a64dd699b1b3b52dac")
    elseif((PYTHON_VERSION MATCHES "3.7") OR (PYTHON_VERSION MATCHES "3.8"))
        set(REQ_URL "https://gitee.com/mirrors/opencv/repository/archive/4.2.0.tar.gz")
        set(MD5 "a60c420b4f0e4a4a7b570995367fc443")
    else()
        message("Could not find 'Python 3.8' or 'Python 3.7' or 'Python 3.9'")
        return()
    endif()
else()
    if("${BUILD_LITE}" STREQUAL "on" OR PYTHON_VERSION MATCHES "3.9")
        set(REQ_URL "https://github.com/opencv/opencv/archive/4.5.1.tar.gz")
        set(MD5 "2205d3169238ec1f184438a96de68513")
    elseif((PYTHON_VERSION MATCHES "3.7") OR (PYTHON_VERSION MATCHES "3.8"))
        set(REQ_URL "https://github.com/opencv/opencv/archive/4.2.0.tar.gz")
        set(MD5 "e8cb208ce2723481408b604b480183b6")
    else()
        message("Could not find 'Python 3.8' or 'Python 3.7' or 'Python 3.9'")
        return()
    endif()
endif()

if(MSVC)
    mindspore_add_pkg(opencv
            VER 4.5.1
            LIBS opencv_core451.lib opencv_imgcodecs451.lib opencv_imgproc451.lib
            LIB_PATH x64/*/lib
            URL ${REQ_URL}
            MD5 ${MD5}
            CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
            -DWITH_ADE=OFF
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
elseif(WIN32)
    if("${BUILD_LITE}" STREQUAL "on" OR PYTHON_VERSION MATCHES "3.9")
        mindspore_add_pkg(opencv
                VER 4.5.1
                LIBS libopencv_core451.dll.a libopencv_imgcodecs451.dll.a libopencv_imgproc451.dll.a
                LIB_PATH x64/mingw/lib
                URL ${REQ_URL}
                MD5 ${MD5}
                CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
                -DWITH_ADE=OFF
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
    elseif(PYTHON_VERSION MATCHES "3.8" OR PYTHON_VERSION MATCHES "3.7")
        mindspore_add_pkg(opencv
                VER 4.2.0
                LIBS libopencv_core420.dll.a libopencv_imgcodecs420.dll.a libopencv_imgproc420.dll.a
                LIB_PATH x64/mingw/lib
                URL ${REQ_URL}
                MD5 ${MD5}
                CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
                -DWITH_ADE=OFF
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
                -DWITH_LAPACK=OFF
                -DTIFF_INCLUDE_DIR=${tiff_INC}
                -DTIFF_LIBRARY=${tiff_LIB})
    endif()
else()
    if("${BUILD_LITE}" STREQUAL "on"  OR PYTHON_VERSION MATCHES "3.9")
        mindspore_add_pkg(opencv
                VER 4.5.1
                LIBS opencv_core opencv_imgcodecs opencv_imgproc
                URL ${REQ_URL}
                MD5  ${MD5}
                CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
                -DWITH_ADE=OFF
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
    elseif(PYTHON_VERSION MATCHES "3.8" OR PYTHON_VERSION MATCHES "3.7")
        mindspore_add_pkg(opencv
                VER 4.2.0
                LIBS opencv_core opencv_imgcodecs opencv_imgproc
                URL ${REQ_URL}
                MD5  ${MD5}
                CMAKE_OPTION -DCMAKE_BUILD_TYPE=Release -DWITH_PROTOBUF=OFF -DWITH_WEBP=OFF -DWITH_IPP=OFF
                -DWITH_ADE=OFF
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
                -DWITH_LAPACK=OFF
                -DTIFF_INCLUDE_DIR=${tiff_INC}
                -DTIFF_LIBRARY=${tiff_LIB})
    endif()
endif()

if(MSVC)
    include_directories(${opencv_INC})
    add_library(mindspore::opencv_core ALIAS opencv::opencv_core451.lib)
    add_library(mindspore::opencv_imgcodecs ALIAS opencv::opencv_imgcodecs451.lib)
    add_library(mindspore::opencv_imgproc ALIAS opencv::opencv_imgproc451.lib)
elseif(WIN32)
    if("${BUILD_LITE}" STREQUAL "on" OR PYTHON_VERSION MATCHES "3.9")
        include_directories(${opencv_INC})
        add_library(mindspore::opencv_core ALIAS opencv::libopencv_core451.dll.a)
        add_library(mindspore::opencv_imgcodecs ALIAS opencv::libopencv_imgcodecs451.dll.a)
        add_library(mindspore::opencv_imgproc ALIAS opencv::libopencv_imgproc451.dll.a)
    elseif(PYTHON_VERSION MATCHES "3.8" OR PYTHON_VERSION MATCHES "3.7")
        include_directories(${opencv_INC})
        add_library(mindspore::opencv_core ALIAS opencv::libopencv_core420.dll.a)
        add_library(mindspore::opencv_imgcodecs ALIAS opencv::libopencv_imgcodecs420.dll.a)
        add_library(mindspore::opencv_imgproc ALIAS opencv::libopencv_imgproc420.dll.a)
    endif()
else()
    include_directories(${opencv_INC}/opencv4)
    add_library(mindspore::opencv_core ALIAS opencv::opencv_core)
    add_library(mindspore::opencv_imgcodecs ALIAS opencv::opencv_imgcodecs)
    add_library(mindspore::opencv_imgproc ALIAS opencv::opencv_imgproc)
endif()
