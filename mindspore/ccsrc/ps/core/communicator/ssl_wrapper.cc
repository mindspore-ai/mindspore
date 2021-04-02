
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

#include "ps/core/communicator/ssl_wrapper.h"

namespace mindspore {
namespace ps {
namespace core {
SSLWrapper::SSLWrapper() : ssl_ctx_(nullptr) { InitSSL(); }

SSLWrapper::~SSLWrapper() { CleanSSL(); }

void SSLWrapper::InitSSL() {
  SSL_library_init();
  ERR_load_crypto_strings();
  SSL_load_error_strings();
  OpenSSL_add_all_algorithms();
  int rand = RAND_poll();
  if (rand == 0) {
    MS_LOG(ERROR) << "RAND_poll failed";
  }
  ssl_ctx_ = SSL_CTX_new(SSLv23_server_method());
  if (!ssl_ctx_) {
    MS_LOG(ERROR) << "SSL_CTX_new failed";
  }
  X509_STORE *store = SSL_CTX_get_cert_store(ssl_ctx_);
  if (X509_STORE_set_default_paths(store) != 1) {
    MS_LOG(ERROR) << "X509_STORE_set_default_paths failed";
  }
}

void SSLWrapper::CleanSSL() {
  if (ssl_ctx_ != nullptr) {
    SSL_CTX_free(ssl_ctx_);
  }
  ERR_free_strings();
  EVP_cleanup();
  ERR_remove_thread_state(nullptr);
  CRYPTO_cleanup_all_ex_data();
}

SSL_CTX *SSLWrapper::GetSSLCtx() { return ssl_ctx_; }
}  // namespace core
}  // namespace ps
}  // namespace mindspore
