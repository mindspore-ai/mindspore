if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/ps-lite/repository/archive/34fd45cae457d59850fdcb2066467778d0673f21.zip")
    set(MD5 "0d1543b8dcb0bc3610637e1643c94eb4")
else()
    set(REQ_URL "https://github.com/dmlc/ps-lite/archive/34fd45cae457d59850fdcb2066467778d0673f21.zip")
    set(MD5 "393c0e27b68bfaf96718caa3aa96f5a3")
endif()

set(pslite_USE_STATIC_LIBS ON)
if(${ENABLE_IBVERBS} STREQUAL "ON")
    set(pslite_CXXFLAGS "USE_IBVERBS=1")
endif()
mindspore_add_pkg(pslite
        LIBS ps
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/pslite/ps_lite.patch001
        ONLY_MAKE True
        ONLY_MAKE_INCS include/*
        ONLY_MAKE_LIBS build/*)
include_directories(${pslite_INC})
add_library(mindspore::pslite ALIAS pslite::ps)
