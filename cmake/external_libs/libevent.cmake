set(openssl_USE_STATIC_LIBS ON)
set(libevent_CFLAGS "-fPIC -fvisibility=hidden -fstack-protector-all -D_FORTIFY_SOURCE=2 -O2")
if(NOT CMAKE_SYSTEM_NAME MATCHES "Darwin")
    set(libevent_LDFLAGS "-Wl,-z,now")
endif()

if(NOT MINDSPORE_PROJECT_DIR)
set(MINDSPORE_PROJECT_DIR ${CMAKE_SOURCE_DIR})
endif()

if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
    set(REQ_URL "https://gitee.com/mirrors/libevent/repository/archive/release-2.1.12-stable.tar.gz")
    set(SHA256 "7180a979aaa7000e1264da484f712d403fcf7679b1e9212c4e3d09f5c93efc24")
else()
    set(REQ_URL
      "https://github.com/libevent/libevent/releases/download/release-2.1.12-stable/libevent-2.1.12-stable.tar.gz")
    set(SHA256 "92e6de1be9ec176428fd2367677e61ceffc2ee1cb119035037a27d346b0403bb")
endif()

message("libevent using openssl stub dir: " ${openssl_ROOT})

mindspore_add_pkg(libevent
        VER 2.1.12
        LIBS event event_pthreads event_core event_openssl
        URL ${REQ_URL}
        SHA256 ${SHA256}
        PATCHES ${MINDSPORE_PROJECT_DIR}/third_party/patch/libevent/libevent.patch001
        CMAKE_OPTION -DCMAKE_BUILD_TYPE:STRING=Release -DBUILD_TESTING=OFF -DOPENSSL_ROOT_DIR:PATH=${openssl_ROOT}
          -DEVENT__LIBRARY_TYPE:STRING=STATIC)

include_directories(${libevent_INC})

add_library(mindspore::event ALIAS libevent::event)
add_library(mindspore::event_pthreads ALIAS libevent::event_pthreads)
add_library(mindspore::event_core ALIAS libevent::event_core)
add_library(mindspore::event_openssl ALIAS libevent::event_openssl)
