set(PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

if(ENABLE_GITEE)
    if(PYTHON_VERSION MATCHES "3.9")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
        set(MD5 "a9b7642031f35daf33a75fe837b3dd31")
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
        set(MD5 "a9b7642031f35daf33a75fe837b3dd31")
    elseif(PYTHON_VERSION MATCHES "3.7")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.4.3.tar.gz")
        set(MD5 "b473a37987ce456ea8cc7aab3f9486f9")
    else()
        message("Could not find 'Python 3.8' or 'Python 3.7' or 'Python 3.9'")
        return()
    endif()
else()
    if(PYTHON_VERSION MATCHES "3.9")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz")
        set(MD5 "32a7811f3db423df4ebfc731a28e5901")
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz")
        set(MD5 "32a7811f3db423df4ebfc731a28e5901")
    elseif(PYTHON_VERSION MATCHES "3.7")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz")
        set(MD5 "62254c40f89925bb894be421fe4cdef2")
    else()
        message("Could not find 'Python 3.8' or 'Python 3.7' or 'Python 3.9'")
        return()
    endif()
endif()
set(pybind11_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(pybind11_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

if(PYTHON_VERSION MATCHES "3.9")
    mindspore_add_pkg(pybind11
        VER 2.6.1
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
        )
elseif(PYTHON_VERSION MATCHES "3.8")
    mindspore_add_pkg(pybind11
        VER 2.6.1
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
        )
else()
    mindspore_add_pkg(pybind11
        VER 2.4.3
        URL ${REQ_URL}
        MD5 ${MD5}
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
        )
endif()

include_directories(${pybind11_INC})
find_package(pybind11 REQUIRED)
set_property(TARGET pybind11::module PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::pybind11_module ALIAS pybind11::module)
