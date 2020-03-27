set(pybind11_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(pybind11_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
mindspore_add_pkg(pybind11
        VER 2.4.3
        URL https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz
        MD5 62254c40f89925bb894be421fe4cdef2
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
        )
include_directories(${pybind11_INC})
find_package(pybind11 REQUIRED)
set_property(TARGET pybind11::module PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::pybind11_module ALIAS pybind11::module)
