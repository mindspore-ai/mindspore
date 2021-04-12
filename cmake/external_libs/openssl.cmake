if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/openssl/repository/archive/OpenSSL_1_1_1k.tar.gz")
  set(MD5 "d4acbcc4a5e6c31d86ede95b5d22f7a0")
else()
  set(REQ_URL "https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1k.tar.gz")
  set(MD5 "bdd51a68ad74618dd2519da8e0bcc759")
endif()
mindspore_add_pkg(openssl
  VER 1.1.0
  LIBS ssl crypto
  URL ${REQ_URL}
  MD5 ${MD5}
  CONFIGURE_COMMAND ./config no-zlib no-shared)
include_directories(${openssl_INC})