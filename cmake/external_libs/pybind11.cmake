set(PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

if(NOT PYTHON_VERSION MATCHES "^3\\.(9|10|11|12)$")
    message("Could not find Python versions 3.9 - 3.12")
    return()
endif()

# Unified pybind11 version (>= 2.12 required for NumPy 2.x PyArray_Descr layout).
set(PYBIND_VERSION 2.13.1)
set(PYBIND_DISABLE_GIL_CHECK TRUE)
# Keep pybind11 <= 2.6 compatible behavior where py::str accepts Python bytes.
add_compile_definitions(PYBIND11_STR_LEGACY_PERMISSIVE)
# This patch reverts a PR introduced in pybind 2.6.2, as it causes many runtime exceptions in mindspore
# due to the type checking introduced in this PR.
# Refer to: https://github.com/pybind/pybind11/pull/2701
# And: https://pybind11.readthedocs.io/en/stable/changelog.html#v2-6-2-jan-26-2021
set(PYBIND11_PATCH_FILE "pybind11.patch002")

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.13.1.tar.gz")
    set(SHA256 "2200dda5c64ece586f537af8fd292103a3042cb40c443dde1b70fd1e419d8cb0")
else()
    set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.13.1.tar.gz")
    set(SHA256 "51631e88960a8856f9c497027f55c9f2f9115cafb08c0005439838a05ba17bfc")
endif()

set(pybind11_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(pybind11_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")

if(PYBIND_DISABLE_GIL_CHECK)
    message(WARNING "Macro PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF is added, "
                    "so pybind11's GIL check for py::handle inc_ref() and dec_ref() is disabled.")
    add_definitions(-DPYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF)
endif()

if(ENABLE_TESTCASES)
    add_definitions(-DPYBIND11_NAMESPACE=pybind11)
endif()

mindspore_add_pkg(pybind11
    VER ${PYBIND_VERSION}
    URL ${REQ_URL}
    SHA256 ${SHA256}
    PATCHES ${TOP_DIR}/third_party/patch/pybind11/${PYBIND11_PATCH_FILE}
    CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
)

include_directories(${pybind11_INC})
find_package(pybind11 REQUIRED)
set_property(TARGET pybind11::module PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::pybind11_module ALIAS pybind11::module)
