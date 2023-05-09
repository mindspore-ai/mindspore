set(PYTHON_VERSION ${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR})

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    if(PYTHON_VERSION MATCHES "3.7")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.4.3.tar.gz")
        set(SHA256 "182cf9e2c5a7ae6f03f84cf17e826d7aa2b02aa2f3705db684dfe686c0278b36")
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
        set(SHA256 "c840509be94ac97216c3b4a3ed9f3fdba9948dbe38c16fcfaee3acc6dc93ed0e")
    elseif(PYTHON_VERSION MATCHES "3.9")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
        set(SHA256 "c840509be94ac97216c3b4a3ed9f3fdba9948dbe38c16fcfaee3acc6dc93ed0e")
    elseif(PYTHON_VERSION MATCHES "3.10")
        set(REQ_URL "https://gitee.com/mirrors/pybind11/repository/archive/v2.6.1.tar.gz")
        set(SHA256 "c840509be94ac97216c3b4a3ed9f3fdba9948dbe38c16fcfaee3acc6dc93ed0e")
    else()
        message("Could not find 'Python 3.7' or 'Python 3.8' or 'Python 3.9' or 'Python 3.10'")
        return()
    endif()
else()
    if(PYTHON_VERSION MATCHES "3.7")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.4.3.tar.gz")
        set(SHA256 "1eed57bc6863190e35637290f97a20c81cfe4d9090ac0a24f3bbf08f265eb71d")
    elseif(PYTHON_VERSION MATCHES "3.8")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz")
        set(SHA256 "cdbe326d357f18b83d10322ba202d69f11b2f49e2d87ade0dc2be0c5c34f8e2a")
    elseif(PYTHON_VERSION MATCHES "3.9")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz")
        set(SHA256 "cdbe326d357f18b83d10322ba202d69f11b2f49e2d87ade0dc2be0c5c34f8e2a")
    elseif(PYTHON_VERSION MATCHES "3.10")
        set(REQ_URL "https://github.com/pybind/pybind11/archive/v2.6.1.tar.gz")
        set(SHA256 "cdbe326d357f18b83d10322ba202d69f11b2f49e2d87ade0dc2be0c5c34f8e2a")
    else()
        message("Could not find 'Python 3.7' or 'Python 3.8' or 'Python 3.9' or 'Python 3.10'")
        return()
    endif()
endif()
set(pybind11_CXXFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(pybind11_CFLAGS "-D_FORTIFY_SOURCE=2 -O2")
set(pybind11_patch ${TOP_DIR}/third_party/patch/pybind11/pybind11.patch001)

if(PYTHON_VERSION MATCHES "3.7")
    mindspore_add_pkg(pybind11
        VER 2.4.3
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
        )
elseif(PYTHON_VERSION MATCHES "3.8")
    mindspore_add_pkg(pybind11
        VER 2.6.1
        URL ${REQ_URL}
        SHA256 ${SHA256}
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
        )
else()
    mindspore_add_pkg(pybind11
        VER 2.6.1
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${pybind11_patch}
        CMAKE_OPTION -DPYBIND11_TEST=OFF -DPYBIND11_LTO_CXX_FLAGS=FALSE
        )
endif()

include_directories(${pybind11_INC})
find_package(pybind11 REQUIRED)
set_property(TARGET pybind11::module PROPERTY IMPORTED_GLOBAL TRUE)
add_library(mindspore::pybind11_module ALIAS pybind11::module)
