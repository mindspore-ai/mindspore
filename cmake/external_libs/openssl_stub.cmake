if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/openssl/repository/archive/OpenSSL_1_1_0l.tar.gz")
  set(MD5 "9d18479e0cac8ff62f7e3df3cceb69dc")
else()
  set(REQ_URL "https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_0l.tar.gz")
  set(MD5 "46d9a2a92fd39198501503b40954e6f0")
endif()
mindspore_add_pkg(openssl
  VER 1.1.0
  LIBS ssl crypto
  URL ${REQ_URL}
  MD5 ${MD5}
  PATCHES ${CMAKE_SOURCE_DIR}/third_party/patch/openssl-stub/openssl-stub.patch001
  CONFIGURE_COMMAND ./config no-zlib)

include_directories(${openssl_INC})
add_library(mindspore::ssl ALIAS openssl::ssl)
add_library(mindspore::crypto ALIAS openssl::crypto)