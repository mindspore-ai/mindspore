set(OPENSSL_USE_STATIC_LIBS ON)
set(libevent_CFLAGS "-fPIC -fvisibility=hidden -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(libevent_LDFLAGS "-Wl,-z,now")
endif()

if(NOT MINDSPORE_PROJECT_DIR)
set(MINDSPORE_PROJECT_DIR ${CMAKE_SOURCE_DIR})
endif()

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/libevent/repository/archive/release-2.1.13-stable.tar.gz")
    set(SHA256 "1a0885e17dc78afbaeddf13cf849f9238bbc24acdc178464a0d1934d7c5ffbd5")
else()
    set(REQ_URL
      "https://github.com/libevent/libevent/releases/download/release-2.1.13-stable/libevent-2.1.13-stable.tar.gz")
    set(SHA256 "f7e9383b8c0baa81b687e5b5eecc01beefaf1b19b64151d95ed61647fe7a315c")
endif()

message("libevent using openssl stub dir: " ${openssl_ROOT})

# CVE-2026-63380 and CVE-2026-63495 affect the evws WebSocket module
# introduced in the 2.2.x line; libevent 2.1.13 does not ship evws.
mindspore_add_pkg(libevent
        VER 2.1.13
        LIBS event event_pthreads event_core event_openssl
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/libevent/libevent.patch001
                ${MINDSPORE_PROJECT_DIR}/third_party/patch/libevent/libevent.patch002
                ${MINDSPORE_PROJECT_DIR}/third_party/patch/libevent/libevent-cve-2026-ws.patch
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release
          -DEVENT__DISABLE_BENCHMARK:BOOL=ON
          -DEVENT__DISABLE_SAMPLES:BOOL=ON
          -DEVENT__DISABLE_TESTS:BOOL=ON
          -DEVENT__DISABLE_OPENSSL:BOOL=OFF
          -DOPENSSL_ROOT_DIR:PATH=${openssl_ROOT}
          -DOPENSSL_USE_STATIC_LIBS:BOOL=ON
          -DEVENT__LIBRARY_TYPE:STRING=STATIC)

include_directories(${libevent_INC})

add_library(mindspore::event ALIAS libevent::event)
add_library(mindspore::event_pthreads ALIAS libevent::event_pthreads)
add_library(mindspore::event_core ALIAS libevent::event_core)
add_library(mindspore::event_openssl ALIAS libevent::event_openssl)
