set(REQ_URL "https://github.com/sqlite/sqlite/archive/version-3.46.1.tar.gz")
set(SHA256 "99c578c9326b12374a64dedae88a63d17557b5d2b0ac65122be67cb3fa2703da")

if(WIN32)
    if(MSVC)
        mindspore_add_pkg(sqlite
            VER 3.46.1
            LIBS sqlite3
            URL https://sqlite.org/2024/sqlite-amalgamation-3460100.zip
            SHA256 77823cb110929c2bcb0f5d48e4833b5c59a8a6e40cdea3936b99e199dbbe5784
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/sqlite_windows_msvc.patch
            CMAKE_OPTION " "
        )
    else()
        mindspore_add_pkg(sqlite
            VER 3.46.1
            LIBS sqlite3
            URL https://sqlite.org/2024/sqlite-amalgamation-3460100.zip
            SHA256 77823cb110929c2bcb0f5d48e4833b5c59a8a6e40cdea3936b99e199dbbe5784
            PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/sqlite_windows.patch
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
        VER 3.46.1
        LIBS sqlite3
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/CVE-2025-29088.patch
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/CVE-2025-3277.patch
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/CVE-2025-6965.patch
        PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/sqlite/CVE-2025-7709.patch
        CONFIGURE_COMMAND ./configure --enable-shared=no --disable-tcl --disable-editline --enable-json1)
endif()

include_directories(${sqlite_INC})
add_library(mindspore::sqlite ALIAS sqlite::sqlite3)
