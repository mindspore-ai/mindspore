set(LIB_ICU_COMMON icuuc)
set(LIB_ICU_DATA icudata)
set(LIB_ICU_I18N icui18n)

if(ENABLE_GITEE_EULER)
    set(GIT_REPOSITORY "https://gitee.com/src-openeuler/icu.git")
    set(GIT_TAG "openEuler-22.03-LTS")
    set(SHA256 "612513798f36d229f7467cc8f0aef4dfb16c7012e1f4d8dbeee813922c4dfe9f")
    set(ICU4C_SRC "${TOP_DIR}/build/mindspore/_deps/icu4c-src/icu4c")
    set(ICU4C_TAR_SRC "${TOP_DIR}/build/mindspore/_deps/icu4c-src")
    __download_pkg_with_git(icu4c ${GIT_REPOSITORY} ${GIT_TAG} ${SHA256})
    execute_process(COMMAND mkdir ${ICU4C_SRC})
    execute_process(COMMAND tar -xf ${ICU4C_TAR_SRC}/icu4c-69_1-src.tgz --strip-components 1 -C ${ICU4C_SRC})
else()
if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/icu/repository/archive/release-69-1.tar.gz")
    set(SHA256 "cb085ebf0152045a84c84011348bb7e09e38248c02b5a85f69a7d85866774b82")
else()
    set(REQ_URL "https://github.com/unicode-org/icu/archive/release-69-1.tar.gz")
    set(SHA256 "39ce83dd5d15c7539dde261733e106a391923f82caf1ce52ecaebb72d93b4579")
endif()
endif()

if(CMAKE_SYSTEM_NAME MATCHES "Windows")
    message("icu4c thirdparty do not support windows currently.")
else()
    set(JSON_FILE "{ \n\
  \"strategy\": \"additive\",\n\
  \"featureFilters\": {\n\
    \"normalization\": \"include\"\n\
  }\n\
}\
    ")
    file(WRITE ${CMAKE_BINARY_DIR}/icu4c_filter.json ${JSON_FILE})
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        mindspore_add_pkg(icu4c
                VER 69.1
                LIBS ${LIB_ICU_COMMON} ${LIB_ICU_DATA} ${LIB_ICU_I18N}
                URL ${REQ_URL}
                SHA256 ${SHA256}
                PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/icu4c/icu4c.patch01
                CONFIGURE_COMMAND ./icu4c/source/runConfigureICU MacOSX --disable-tests
                                  --disable-samples --disable-icuio --disable-extras
                                  ICU_DATA_FILTER_FILE=${CMAKE_BINARY_DIR}/icu4c_filter.json
                )
    else()
        mindspore_add_pkg(icu4c
                VER 69.1
                LIBS ${LIB_ICU_COMMON} ${LIB_ICU_DATA} ${LIB_ICU_I18N}
                URL ${REQ_URL}
                SHA256 ${SHA256}
                PATCHES ${TOP_DIR}/third_party/patch/icu4c/icu4c.patch01
                CONFIGURE_COMMAND ./icu4c/source/runConfigureICU Linux --enable-rpath --disable-tests --disable-samples
                                  --disable-icuio --disable-extras
                                  ICU_DATA_FILTER_FILE=${CMAKE_BINARY_DIR}/icu4c_filter.json
                )
    endif()
    include_directories(${icu4c_INC})
    if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
        include(${CMAKE_SOURCE_DIR}/cmake/change_rpath.cmake)
        changerpath($<TARGET_FILE:icu4c::${LIB_ICU_COMMON}> ${LIB_ICU_COMMON} "libicuuc;libicudata")
        changerpath($<TARGET_FILE:icu4c::${LIB_ICU_DATA}> ${LIB_ICU_DATA} "libicudata")
        changerpath($<TARGET_FILE:icu4c::${LIB_ICU_I18N}> ${LIB_ICU_I18N} "libicuuc;libicudata;libicui18n")
    endif()
    add_library(mindspore::icuuc ALIAS icu4c::${LIB_ICU_COMMON})
    add_library(mindspore::icudata ALIAS icu4c::${LIB_ICU_DATA})
    add_library(mindspore::icui18n ALIAS icu4c::${LIB_ICU_I18N})
    add_definitions(-D ENABLE_ICU4C)
endif()
