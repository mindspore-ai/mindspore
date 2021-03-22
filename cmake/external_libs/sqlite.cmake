if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/sqlite/repository/archive/version-3.32.2.tar.gz")
    set(MD5 "7312cad1739d8a73b14abddc850c0afa")
else()
    set(REQ_URL "https://github.com/sqlite/sqlite/archive/version-3.32.2.tar.gz")
    set(MD5 "ea6d3b3289b4ac216fb06081a01ef101")
endif()


if(WIN32)
    mindspore_add_pkg(sqlite
        VER 3.32.2
        LIBS sqlite3
        URL https://sqlite.org/2020/sqlite-amalgamation-3320200.zip
        MD5 1eccea18d248eb34c7378b2b3f63f1db
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/sqlite.windows.patch001
        CMAKE_OPTION " "
    )

else()
    set(sqlite_USE_STATIC_LIBS ON)
    set(sqlite_CXXFLAGS)
    if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
        set(sqlite_CFLAGS "-fstack-protector-all -Wno-uninitialized -Wno-unused-parameter -fPIC -D_FORTIFY_SOURCE=2 \
          -O2")
    else()
        set(sqlite_CFLAGS "-fstack-protector-all -Wno-maybe-uninitialized -Wno-unused-parameter -fPIC \
          -D_FORTIFY_SOURCE=2 -O2")
        set(sqlite_LDFLAGS "-Wl,-z,relro,-z,now,-z,noexecstack")
    endif()
    mindspore_add_pkg(sqlite
        VER 3.32.2
        LIBS sqlite3
        URL ${REQ_URL}
        MD5 ${MD5}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/sqlite.patch001
        CONFIGURE_COMMAND ./configure --enable-shared=no --disable-tcl --disable-editline --enable-json1)
endif()

include_directories(${sqlite_INC})
add_library(mindspore::sqlite ALIAS sqlite::sqlite3)
