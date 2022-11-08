if(ENABLE_GITEE)
    set(REQ_URL "https://gitee.com/mirrors/sqlite/repository/archive/version-3.36.0.tar.gz")
    set(SHA256 "a0989fc6e890ac1b1b28661490636617154da064b6bfe6c71100d23a9e7298fd")
else()
    set(REQ_URL "https://github.com/sqlite/sqlite/archive/version-3.36.0.tar.gz")
    set(SHA256 "a0989fc6e890ac1b1b28661490636617154da064b6bfe6c71100d23a9e7298fd")
endif()


if(WIN32)
    if(MSVC)
        mindspore_add_pkg(sqlite
            VER 3.36.0
            LIBS sqlite3
            URL https://sqlite.org/2021/sqlite-amalgamation-3360000.zip
            SHA256 999826fe4c871f18919fdb8ed7ec9dd8217180854dd1fe21eea96aed36186729
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/sqlite.windows.msvc.patch001
            CMAKE_OPTION " "
        )
    else()
        mindspore_add_pkg(sqlite
            VER 3.36.0
            LIBS sqlite3
            URL https://sqlite.org/2021/sqlite-amalgamation-3360000.zip
            SHA256 999826fe4c871f18919fdb8ed7ec9dd8217180854dd1fe21eea96aed36186729
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/sqlite.windows.patch002
            CMAKE_OPTION " "
        )
    endif()
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
        VER 3.36.0
        LIBS sqlite3
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/CVE-2022-35737.patch
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/CVE-2021-36690.patch
        CONFIGURE_COMMAND ./configure --enable-shared=no --disable-tcl --disable-editline --enable-json1)
endif()

include_directories(${sqlite_INC})
add_library(mindspore::sqlite ALIAS sqlite::sqlite3)
