if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
  set(REQ_URL "https://gitee.com/mirrors/openssl/repository/archive/openssl-3.5.4.tar.gz")
  set(SHA256 "758b69feed5787dc12d34b0eb29b60d3c9d73d5a64760c62d93a6d26b344d65d")
else()
  set(REQ_URL "https://github.com/openssl/openssl/archive/refs/tags/openssl-3.5.4.tar.gz")
  set(SHA256 "758b69feed5787dc12d34b0eb29b60d3c9d73d5a64760c62d93a6d26b344d65d")
endif()

set(OPENSSL_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/openssl)

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE)
    set(openssl_CFLAGS -fvisibility=hidden)
    mindspore_add_pkg(openssl
            VER 3.5.4
            LIBS ssl crypto
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CONFIGURE_COMMAND ./config --libdir=lib no-zlib no-shared no-afalgeng no-tests no-docs
            )
    include_directories(${openssl_INC})
    add_library(mindspore::ssl ALIAS openssl::ssl)
    add_library(mindspore::crypto ALIAS openssl::crypto)
endif()
