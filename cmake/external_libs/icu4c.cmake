set(LIB_ICU_COMMON icuuc)
set(LIB_ICU_DATA icudata)
set(LIB_ICU_I18N icui18n)
if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    message("icu4c thirdparty do not support windows currently.")
else()
    mindspore_add_pkg(icu4c
            VER 67.1
            LIBS ${LIB_ICU_COMMON} ${LIB_ICU_DATA} ${LIB_ICU_I18N}
            URL https://github.com/unicode-org/icu/archive/release-67-1.tar.gz
            MD5 0c2662a2b0bc80b0eb56495205247c8f
            CONFIGURE_COMMAND ./icu4c/source/runConfigureICU Linux --enable-rpath --disable-tests --disable-samples --disable-icuio --disable-extras ICU_DATA_FILTER_FILE=${CMAKE_SOURCE_DIR}/third_party/icu4c/filter.json
            )
    include_directories(${icu4c_INC})
    add_library(mindspore::icuuc ALIAS icu4c::${LIB_ICU_COMMON})
    add_library(mindspore::icudata ALIAS icu4c::${LIB_ICU_DATA})
    add_library(mindspore::icui18n ALIAS icu4c::${LIB_ICU_I18N})
    add_definitions(-D ENABLE_ICU4C)
endif()