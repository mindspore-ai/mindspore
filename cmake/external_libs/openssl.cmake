if(ENABLE_GITEE)
  set(REQ_URL "https://gitee.com/mirrors/openssl/repository/archive/OpenSSL_1_1_1k.tar.gz")
  set(MD5 "5e96e1713bb1f93358f68cf1a85d4512")
else()
  set(REQ_URL "https://github.com/openssl/openssl/archive/refs/tags/OpenSSL_1_1_1k.tar.gz")
  set(MD5 "bdd51a68ad74618dd2519da8e0bcc759")
endif()

if(BUILD_LITE)
  set(OPENSSL_PATCH_ROOT ${TOP_DIR}/third_party/patch/openssl)
else()
  set(OPENSSL_PATCH_ROOT ${CMAKE_SOURCE_DIR}/third_party/patch/openssl)
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
  mindspore_add_pkg(openssl
    VER 1.1.1k
    LIBS ssl crypto
    URL ${REQ_URL}
    MD5 ${MD5}
    CONFIGURE_COMMAND ./config no-zlib no-shared
    PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3711.patch
    PATCHES ${OPENSSL_PATCH_ROOT}/CVE-2021-3712.patch
  )
  include_directories(${openssl_INC})
  add_library(mindspore::ssl ALIAS openssl::ssl)
  add_library(mindspore::crypto ALIAS openssl::crypto)
endif()
