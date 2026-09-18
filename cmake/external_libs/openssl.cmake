if(ENABLE_GITEE OR ENABLE_GITEE_EULER) # Channel GITEE_EULER is NOT supported now, use GITEE instead.
  set(REQ_URL "https://gitee.com/mirrors/openssl/repository/archive/openssl-3.5.8.tar.gz")
  set(SHA256 "59f86483992995df5a213df38d93f31eeafe3b34599309b3e088ba67ca0aad9c")
else()
  set(REQ_URL "https://github.com/openssl/openssl/archive/refs/tags/openssl-3.5.8.tar.gz")
  set(SHA256 "59f86483992995df5a213df38d93f31eeafe3b34599309b3e088ba67ca0aad9c")
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux" OR APPLE)
    set(openssl_CFLAGS -fvisibility=hidden)
    mindspore_add_pkg(openssl
            VER 3.5.8
            LIBS ssl crypto
            URL ${REQ_URL}
            SHA256 ${SHA256}
            CONFIGURE_COMMAND ./config --libdir=lib no-zlib no-shared no-afalgeng no-tests no-docs
            )
    include_directories(${openssl_INC})
    add_library(mindspore::ssl ALIAS openssl::ssl)
    add_library(mindspore::crypto ALIAS openssl::crypto)
endif()
