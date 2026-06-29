if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
  set(REQ_URL "https://gitee.com/mirrors/openssl/repository/archive/openssl-3.5.7.tar.gz")
  set(SHA256 "d71a811bfbd9153d7b30cbe476263302ee4b04a9a47ffea6e6a782326805c93f")
else()
  set(REQ_URL "https://github.com/openssl/openssl/archive/refs/tags/openssl-3.5.7.tar.gz")
  set(SHA256 "d71a811bfbd9153d7b30cbe476263302ee4b04a9a47ffea6e6a782326805c93f")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE)
    set(openssl_CFLAGS -fvisibility=hidden)
    mindspore_add_pkg(openssl
            VER 3.5.7
            LIBS ssl crypto
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CONFIGURE_COMMAND ./config --libdir=lib no-zlib no-shared no-afalgeng no-tests no-docs
            )
    include_directories(${openssl_INC})
    add_library(mindspore::ssl ALIAS openssl::ssl)
    add_library(mindspore::crypto ALIAS openssl::crypto)
endif()
