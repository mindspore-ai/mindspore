/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_SSL_HTTP_H_
#define MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_SSL_HTTP_H_

#include <openssl/ssl.h>
#include <openssl/rand.h>
#include <openssl/err.h>
#include <openssl/evp.h>
#include <assert.h>
#include <openssl/pkcs12.h>
#include <openssl/bio.h>

#include <iostream>
#include <string>
#include <memory>

#include "utils/log_adapter.h"
#include "ps/core/comm_util.h"
#include "include/backend/distributed/ps/constants.h"
#include "ps/core/file_configuration.h"

namespace mindspore {
namespace ps {
namespace core {
class SSLHTTP {
 public:
  static SSLHTTP &GetInstance() {
    static SSLHTTP instance;
    return instance;
  }
  SSL_CTX *GetSSLCtx() const;

 private:
  SSLHTTP();
  virtual ~SSLHTTP();
  SSLHTTP(const SSLHTTP &) = delete;
  SSLHTTP &operator=(const SSLHTTP &) = delete;

  void InitSSL();
  void CleanSSL();
  void InitSSLCtx(const X509 *cert, const EVP_PKEY *pkey, const std::string &default_cipher_list);
  SSL_CTX *ssl_ctx_;
};
}  // namespace core
}  // namespace ps
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PS_CORE_COMMUNICATOR_SSL_HTTP_H_
