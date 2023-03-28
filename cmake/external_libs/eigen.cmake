set(Eigen3_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(Eigen3_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")


set(REQ_URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz")
set(SHA256 "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72")

if(MSVC)
    mindspore_add_pkg(Eigen3
            VER 3.4.0
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CMAKE_OPTION -DBUILD_TESTING=OFF)
else()
    mindspore_add_pkg(Eigen3
            VER 3.4.0
            URL ${REQ_URL}
            SHA256 ${SHA256}
            PATCHES ${TOP_DIR}/third_party/patch/eigen/0001-fix-eigen.patch
            CMAKE_OPTION -DBUILD_TESTING=OFF)
endif()
find_package(Eigen3 3.4.0 REQUIRED ${MS_FIND_NO_DEFAULT_PATH})
include_directories(${Eigen3_INC})
include_directories(${EIGEN3_INCLUDE_DIR})
set_property(TARGET Eigen3::Eigen PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::eigen ALIAS Eigen3::Eigen)
