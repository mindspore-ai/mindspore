set(PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

set(PYBIND_DISABLE_GIL_CHECK FALSE)
set(PYBIND11_PATCH_FILE "")
if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    if(PYTHON_VERSION MATCHES "3.7")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.4.3.tar.gz")
        set(SHA256 "182cf9e2c5a7ae6f03f84cf17e826d7aa2b02aa2f3705db684dfe686c0278b36")
        set(PYBIND_VERSION 2.4.3)
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
        set(SHA256 "c840509be94ac97216c3b4a3ed9f3fdba9948dbe38c16fcfaee3acc6dc93ed0e")
        set(PYBIND_VERSION 2.6.1)
    elseif(PYTHON_VERSION MATCHES "3.9")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
        set(SHA256 "c840509be94ac97216c3b4a3ed9f3fdba9948dbe38c16fcfaee3acc6dc93ed0e")
        set(PYBIND_VERSION 2.6.1)
        set(PYBIND11_PATCH_FILE "pybind11.patch001")
    elseif(PYTHON_VERSION MATCHES "3.10")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
        set(SHA256 "c840509be94ac97216c3b4a3ed9f3fdba9948dbe38c16fcfaee3acc6dc93ed0e")
        set(PYBIND_VERSION 2.6.1)
        set(PYBIND11_PATCH_FILE "pybind11.patch001")
    elseif(PYTHON_VERSION MATCHES "3.11" OR PYTHON_VERSION MATCHES "3.12")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.13.1.tar.gz")
        set(SHA256 "2200dda5c64ece586f537af8fd292103a3042cb40c443dde1b70fd1e419d8cb0")
        set(PYBIND_VERSION 2.13.1)
        set(PYBIND_DISABLE_GIL_CHECK TRUE)
        # This patch reverts a PR introduced in pybind 2.6.2, as it causes many runtime exceptions in mindspore
        # due to the type checking introduced in this PR.
        # Refer to: https://github.com/pybind/pybind11/pull/2701
        # And: https://pybind11.readthedocs.io/en/stable/changelog.html#v2-6-2-jan-26-2021
        set(PYBIND11_PATCH_FILE "pybind11.patch002")
    else()
        message("Could not find Python versions 3.7 - 3.12")
        return()
    endif()
else()
    if(PYTHON_VERSION MATCHES "3.7")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz")
        set(SHA256 "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d")
        set(PYBIND_VERSION 2.4.3)
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz")
        set(SHA256 "cdbe326d357f18b83d10322ba202d69f11b2f49e2d87ade0dc2be0c5c34f8e2a")
        set(PYBIND_VERSION 2.6.1)
    elseif(PYTHON_VERSION MATCHES "3.9")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz")
        set(SHA256 "cdbe326d357f18b83d10322ba202d69f11b2f49e2d87ade0dc2be0c5c34f8e2a")
        set(PYBIND_VERSION 2.6.1)
        set(PYBIND11_PATCH_FILE "pybind11.patch001")
    elseif(PYTHON_VERSION MATCHES "3.10")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz")
        set(SHA256 "cdbe326d357f18b83d10322ba202d69f11b2f49e2d87ade0dc2be0c5c34f8e2a")
        set(PYBIND_VERSION 2.6.1)
        set(PYBIND11_PATCH_FILE "pybind11.patch001")
    elseif(PYTHON_VERSION MATCHES "3.11" OR PYTHON_VERSION MATCHES "3.12")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.13.1.tar.gz")
        set(SHA256 "51631e88960a8856f9c497027f55c9f2f9115cafb08c0005439838a05ba17bfc")
        set(PYBIND_VERSION 2.13.1)
        set(PYBIND_DISABLE_GIL_CHECK TRUE)
        set(PYBIND11_PATCH_FILE "pybind11.patch002")
    else()
        message("Could not find Python versions 3.7 - 3.12")
        return()
    endif()
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

if(NOT "${PYBIND11_PATCH_FILE}" STREQUAL "")
    mindspore_add_pkg(pybind11
        VER ${PYBIND_VERSION}
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${TOP_DIR}/third_party/patch/pybind11/${PYBIND11_PATCH_FILE}
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
    )
else()
    mindspore_add_pkg(pybind11
        VER ${PYBIND_VERSION}
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
    )
endif()

include_directories(${pybind11_INC})
find_package(pybind11 REQUIRED)
set_property(TARGET pybind11::module PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::pybind11_module ALIAS pybind11::module)
