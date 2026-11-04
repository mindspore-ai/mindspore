set(Eigen3_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(Eigen3_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")


set(REQ_URL "https://gitlab.com/libeigen/eigen/-/archive/5.0.0/eigen-5.0.0.tar.gz")
set(SHA256 "315c881e19e17542a7d428c5aa37d113c89b9500d350c433797b730cd449c056")

if(MSVC)
    mindspore_add_pkg(Eigen3
            VER 5.0.0
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CMAKE_OPTION
                -DBUILD_TESTING=OFF
                -DEIGEN_BUILD_TESTING=OFF
                -DEIGEN_BUILD_BLAS=OFF
                -DEIGEN_BUILD_LAPACK=OFF
                -DEIGEN_BUILD_DEMOS=OFF
                -DEIGEN_BUILD_DOC=OFF
                -DEIGEN_BUILD_BTL=OFF
                -DEIGEN_BUILD_SPBENCH=OFF
                -DEIGEN_BUILD_PKGCONFIG=OFF
                -DEIGEN_BUILD_CMAKE_PACKAGE=ON)
else()
    mindspore_add_pkg(Eigen3
            VER 5.0.0
            URL ${REQ_URL}
            SHA256 ${SHA256}
            PATCHES ${TOP_DIR}/third_party/patch/eigen/0001-fix-eigen.patch
            CMAKE_OPTION
                -DBUILD_TESTING=OFF
                -DEIGEN_BUILD_TESTING=OFF
                -DEIGEN_BUILD_BLAS=OFF
                -DEIGEN_BUILD_LAPACK=OFF
                -DEIGEN_BUILD_DEMOS=OFF
                -DEIGEN_BUILD_DOC=OFF
                -DEIGEN_BUILD_BTL=OFF
                -DEIGEN_BUILD_SPBENCH=OFF
                -DEIGEN_BUILD_PKGCONFIG=OFF
                -DEIGEN_BUILD_CMAKE_PACKAGE=ON)
endif()
find_package(Eigen3 5.0.0 REQUIRED ${MS_FIND_NO_DEFAULT_PATH})
include_directories(SYSTEM ${Eigen3_INC})
include_directories(SYSTEM ${EIGEN3_INCLUDE_DIR})

# NOTE:
# Eigen headers may be installed under:
#   - <prefix>/include/Eigen/...
#   - <prefix>/include/eigen3/Eigen/...
# With Eigen 5, relying only on legacy vars (Eigen3_INC/EIGEN3_INCLUDE_DIR) may miss the real include dir.
# So we also import include dirs from Eigen3::Eigen (INTERFACE_INCLUDE_DIRECTORIES) to fix missing headers
# such as `#include <Eigen/Dense>`.
if(TARGET Eigen3::Eigen)
    get_target_property(_EIGEN3_INTERFACE_INCS Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    if(_EIGEN3_INTERFACE_INCS)
        include_directories(SYSTEM ${_EIGEN3_INTERFACE_INCS})
    endif()
endif()
set_property(TARGET Eigen3::Eigen PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::eigen ALIAS Eigen3::Eigen)
