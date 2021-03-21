set(LIB_ICU_COMMON icuuc)
set(LIB_ICU_DATA icudata)
set(LIB_ICU_I18N icui18n)

if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/icu/repository/archive/release-67-1.tar.gz")
    set(MD5 "72415ffd1af3acf19f9aa3fa82c7b5bc")
else()
    set(REQ_URL "https://github.com/unicode-org/icu/archive/release-67-1.tar.gz")
    set(MD5 "fd525fb47d8827b0b7da78b51dd2d93f")
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
                VER 67.1
                LIBS ${LIB_ICU_COMMON} ${LIB_ICU_DATA} ${LIB_ICU_I18N}
                URL ${REQ_URL}
                MD5 ${MD5}
                PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/icu4c/icu4c.patch01
                CONFIGURE_COMMAND ./icu4c/source/runConfigureICU MacOSX --enable-rpath --disable-tests
                                  --disable-samples --disable-icuio --disable-extras
                                  ICU_DATA_FILTER_FILE=${CMAKE_BINARY_DIR}/icu4c_filter.json
                )
    else()
        mindspore_add_pkg(icu4c
                VER 67.1
                LIBS ${LIB_ICU_COMMON} ${LIB_ICU_DATA} ${LIB_ICU_I18N}
                URL ${REQ_URL}
                MD5 ${MD5}
                PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/icu4c/icu4c.patch01
                CONFIGURE_COMMAND ./icu4c/source/runConfigureICU Linux --enable-rpath --disable-tests --disable-samples
                                  --disable-icuio --disable-extras
                                  ICU_DATA_FILTER_FILE=${CMAKE_BINARY_DIR}/icu4c_filter.json
                )
    endif()
    include_directories(${icu4c_INC})
    add_library(mindspore::icuuc ALIAS icu4c::${LIB_ICU_COMMON})
    add_library(mindspore::icudata ALIAS icu4c::${LIB_ICU_DATA})
    add_library(mindspore::icui18n ALIAS icu4c::${LIB_ICU_I18N})
    add_definitions(-D ENABLE_ICU4C)
endif()
